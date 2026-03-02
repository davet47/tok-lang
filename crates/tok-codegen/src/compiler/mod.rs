/// Cranelift code generation: HIR → native object file.
///
/// The compiler translates `HirProgram` into Cranelift IR and emits a `.o`
/// object file. A subsequent linker step (driven by the CLI) joins it with
/// `libtok_rt.a` to produce an executable.
///
/// # Value representation
///
/// Static types let us keep most values unboxed:
///
/// | Tok Type | Cranelift | CL type |
/// |----------|-----------|---------|
/// | Int      | i64       | I64     |
/// | Float    | f64       | F64     |
/// | Bool     | i8        | I8      |
/// | Nil      | (nothing) | —       |
/// | String   | *mut TokString | I64 (ptr) |
/// | Array    | *mut TokArray  | I64 (ptr) |
/// | Map      | *mut TokMap    | I64 (ptr) |
/// | Tuple    | *mut TokTuple  | I64 (ptr) |
/// | Closure  | *mut TokClosure | I64 (ptr) |
/// | Channel  | *mut TokChannel | I64 (ptr) |
/// | Handle   | *mut TokHandle  | I64 (ptr) |
/// | Any      | TokValue (tag:i8 + pad + data:i64) = 16 bytes | [I64, I64] |
use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, Block, InstBuilder, SigRef, Value};
use cranelift_codegen::isa;
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule, ObjectProduct};
use target_lexicon::Triple;

use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use tok_hir::hir::*;
use tok_types::Type;

mod stdlib;
pub use stdlib::is_stdlib_module;
use stdlib::{get_stdlib_const, get_stdlib_func, stdlib_constructor};
mod runtime_decls;

// ─── Cranelift type helpers ────────────────────────────────────────────

/// The pointer type on the target (always 64-bit for now).
const PTR: types::Type = types::I64;

/// Map a Tok `Type` to the Cranelift IR type(s) it occupies.
/// Returns None for Nil (zero-size).
fn cl_type(ty: &Type) -> Option<types::Type> {
    match ty {
        Type::Int | Type::Range => Some(types::I64),
        Type::Float => Some(types::F64),
        Type::Bool => Some(types::I8),
        Type::Nil | Type::Never => None,
        // All heap-allocated types are pointers
        Type::Str
        | Type::Array(_)
        | Type::Map(_)
        | Type::Tuple(_)
        | Type::Func(_)
        | Type::Optional(_)
        | Type::Result(_)
        | Type::Channel(_)
        | Type::Handle(_) => Some(PTR),
        // Any = TokValue = 16 bytes. We pass/return as a single I128
        // but store in memory as two I64s (tag + data). For simplicity
        // in the initial implementation, we represent Any as I64 (pointer
        // to stack-allocated TokValue).
        Type::Any => Some(PTR),
    }
}

/// Return the Cranelift type or I64 as default (for things like Nil returns
/// where we still need to produce a value in some contexts).
fn cl_type_or_i64(ty: &Type) -> types::Type {
    cl_type(ty).unwrap_or(types::I64)
}

/// Is this type a heap-allocated pointer that needs refcount management?
fn is_heap_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Str
            | Type::Array(_)
            | Type::Map(_)
            | Type::Tuple(_)
            | Type::Func(_)
            | Type::Channel(_)
            | Type::Handle(_)
            | Type::Optional(_)
            | Type::Result(_)
    )
}

/// Emit a runtime call to decrement the reference count of a value.
/// No-ops for scalar types (Int, Float, Bool, Nil). For heap types and Any,
/// calls tok_value_rc_dec(tag, data) which handles recursive cleanup.
/// Optimized fast path for strings: calls tok_string_free(ptr) directly.
fn emit_rc_dec(ctx: &mut FuncCtx, val: Value, ty: &Type) {
    match ty {
        Type::Int | Type::Float | Type::Bool | Type::Nil | Type::Never => {
            return; // Scalars — no RC needed
        }
        Type::Str => {
            // Fast path: direct string free without TokValue reconstruction
            let func_ref = ctx.get_runtime_func_ref("tok_string_free");
            ctx.builder.ins().call(func_ref, &[val]);
            return;
        }
        _ => {}
    }
    let (tag, data) = to_tokvalue(ctx, val, ty);
    let func_ref = ctx.get_runtime_func_ref("tok_value_rc_dec");
    ctx.builder.ins().call(func_ref, &[tag, data]);
}

// ─── Compiler state ───────────────────────────────────────────────────

/// A captured variable from an enclosing scope, to be stored in the closure environment.
#[derive(Debug, Clone)]
struct CapturedVar {
    name: String,
    ty: Type,
}

/// A lambda that has been declared but not yet compiled (deferred compilation).
struct PendingLambda {
    name: String,
    func_id: FuncId,
    params: Vec<HirParam>,
    ret_type: Type,
    body: Vec<HirStmt>,
    /// Variables captured from the enclosing scope.
    captures: Vec<CapturedVar>,
    /// If set, this lambda is compiled with specialized (native-typed) calling convention.
    /// Contains concrete types for each parameter, inferred from the call site.
    specialized_param_types: Option<Vec<Type>>,
}

/// Info stored for known closures to enable direct and specialized calls.
#[derive(Clone)]
struct KnownClosure {
    func_id: FuncId,
    env_ptr: Value,
    /// Index into pending_lambdas for the uniform version.
    pending_idx: usize,
    /// Specialized FuncId for a specific set of arg types + return type, if created.
    specialized: Option<(FuncId, Vec<Type>, Type)>,
}

/// Top-level compiler that holds the Cranelift module and all metadata.
pub struct Compiler {
    module: ObjectModule,
    /// The default calling convention for this target.
    #[allow(dead_code)]
    call_conv: CallConv,
    /// Cranelift functions that have been declared (name → FuncId).
    declared_funcs: HashMap<String, FuncId>,
    /// Tok function signatures: param types + return type.
    func_sigs: HashMap<String, (Vec<Type>, Type)>,
    /// Runtime extern functions (tok_* → FuncId).
    runtime_funcs: HashMap<String, FuncId>,
    /// String literal data (index → data id, length).
    string_literals: Vec<(cranelift_module::DataId, usize)>,
    /// Counter for unique names (gensym).
    #[allow(dead_code)]
    gensym_counter: u32,
    /// Counter for generating unique lambda function names.
    lambda_counter: u32,
    /// Lambdas waiting to be compiled (deferred until current function finalizes).
    pending_lambdas: Vec<PendingLambda>,
    /// User function bodies, stored for potential inlining.
    func_bodies: HashMap<String, (Vec<HirParam>, Type, Vec<HirStmt>)>,
}

impl Compiler {
    fn new() -> Self {
        let mut settings_builder = settings::builder();
        settings_builder
            .set("opt_level", "speed")
            .expect("codegen: invalid cranelift setting opt_level");
        settings_builder
            .set("is_pic", "true")
            .expect("codegen: invalid cranelift setting is_pic");
        // Use the host triple
        let triple = Triple::from_str(&target_lexicon::HOST.to_string())
            .expect("codegen: unsupported host triple");
        let flags = settings::Flags::new(settings_builder);
        let isa = isa::lookup(triple.clone())
            .expect("codegen: unsupported ISA for host triple")
            .finish(flags)
            .expect("codegen: failed to build ISA");

        let call_conv = isa.default_call_conv();

        let obj_builder =
            ObjectBuilder::new(isa, "tok_output", cranelift_module::default_libcall_names())
                .expect("codegen: failed to create object builder");
        let module = ObjectModule::new(obj_builder);

        Compiler {
            module,
            call_conv,
            declared_funcs: HashMap::new(),
            func_sigs: HashMap::new(),
            runtime_funcs: HashMap::new(),
            string_literals: Vec::new(),
            gensym_counter: 0,
            lambda_counter: 0,
            pending_lambdas: Vec::new(),
            func_bodies: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    fn gensym(&mut self) -> String {
        self.gensym_counter += 1;
        format!("__tok_tmp_{}", self.gensym_counter)
    }

    // ─── Runtime function declaration ──────────────────────────────

    /// Declare an extern "C" runtime function so we can call it.
    fn declare_runtime_func(
        &mut self,
        name: &str,
        params: &[types::Type],
        returns: &[types::Type],
    ) -> FuncId {
        if let Some(&id) = self.runtime_funcs.get(name) {
            return id;
        }
        let mut sig = self.module.make_signature();
        for &p in params {
            sig.params.push(AbiParam::new(p));
        }
        for &r in returns {
            sig.returns.push(AbiParam::new(r));
        }
        let id = self
            .module
            .declare_function(name, Linkage::Import, &sig)
            .expect("codegen: failed to declare runtime function");
        self.runtime_funcs.insert(name.to_string(), id);
        id
    }

    // declare_all_runtime_funcs is in runtime_decls.rs

    /// Declare a string literal as a data object, returning a DataId.
    fn declare_string_data(&mut self, s: &str) -> (cranelift_module::DataId, usize) {
        let name = format!("__tok_str_{}", self.string_literals.len());
        let data_id = self
            .module
            .declare_data(&name, Linkage::Local, false, false)
            .expect("codegen: failed to declare string data");
        let mut desc = DataDescription::new();
        desc.define(s.as_bytes().to_vec().into_boxed_slice());
        self.module
            .define_data(data_id, &desc)
            .expect("codegen: failed to define string data");
        let entry = (data_id, s.len());
        self.string_literals.push(entry);
        entry
    }

    /// Declare a Tok-level function (for forward references, recursion).
    fn declare_tok_func(&mut self, name: &str, params: &[HirParam], ret_type: &Type) -> FuncId {
        if let Some(&id) = self.declared_funcs.get(name) {
            return id;
        }
        let mut sig = self.module.make_signature();
        // use module's default calling convention (set by make_signature)
        for p in params {
            if matches!(p.ty, Type::Any) {
                // Any params: pass TokValue as (tag: I64, data: I64)
                sig.params.push(AbiParam::new(types::I64));
                sig.params.push(AbiParam::new(types::I64));
            } else if let Some(ct) = cl_type(&p.ty) {
                sig.params.push(AbiParam::new(ct));
            }
        }
        if matches!(ret_type, Type::Any) {
            // Any return: return TokValue as (tag: I64, data: I64)
            sig.returns.push(AbiParam::new(types::I64));
            sig.returns.push(AbiParam::new(types::I64));
        } else if let Some(ct) = cl_type(ret_type) {
            sig.returns.push(AbiParam::new(ct));
        }
        let id = self
            .module
            .declare_function(name, Linkage::Local, &sig)
            .expect("codegen: failed to declare user function");
        self.declared_funcs.insert(name.to_string(), id);
        let param_types: Vec<Type> = params.iter().map(|p| p.ty.clone()).collect();
        self.func_sigs
            .insert(name.to_string(), (param_types, ret_type.clone()));
        id
    }
}

// ─── Function-level codegen context ───────────────────────────────────

/// Per-function compilation state.
pub(crate) struct FuncCtx<'a> {
    compiler: &'a mut Compiler,
    builder: FunctionBuilder<'a>,
    /// Variable name → (Variable, Type)
    vars: HashMap<String, (Variable, Type)>,
    /// Next Variable index.
    next_var: usize,
    /// The return block for early returns.
    return_block: Block,
    /// The return variable (if the function has a non-void return).
    #[allow(dead_code)]
    return_var: Option<Variable>,
    /// Loop context stack: (continue_block, break_block)
    loop_stack: Vec<(Block, Block)>,
    /// Whether the current block has been terminated (by return/break/continue/jump).
    block_terminated: bool,
    /// Whether this function returns Any type (uses 2-value return: tag, data).
    is_any_return: bool,
    /// The return type of the current function.
    #[allow(dead_code)]
    ret_type: Type,
    /// Closures assigned to local variables where we know the FuncId at compile time.
    known_closures: HashMap<String, KnownClosure>,
    /// Variables that hold stdlib module imports: var_name → module_name (e.g. "m" → "math").
    stdlib_imports: HashMap<String, String>,
    /// Set by Lambda compilation so the enclosing Assign can record it in known_closures.
    last_lambda_info: Option<(FuncId, Value, usize)>, // (func_id, env_ptr, pending_idx)
    /// Parameter names (should not be RC dec'd at function exit — caller owns them).
    param_names: HashSet<String>,
    /// Cached SigRef for indirect closure calls, keyed by arg count.
    closure_sig_cache: HashMap<usize, SigRef>,
    /// TCO: function name if this function uses tail-call optimization.
    tco_func_name: Option<String>,
    /// TCO: the loop header block to jump back to for tail calls.
    tco_loop_header: Option<Block>,
    /// TCO: the parameter variables in order (for reassignment on tail call jump).
    tco_param_vars: Vec<Variable>,
}

/// Compile an HIR program to a native object file (bytes).
///
/// # Panics
///
/// Panics on invalid HIR that should have been caught by earlier pipeline
/// stages (lexer, parser, type checker). These are intentional `expect()`
/// and `panic!()` calls for genuinely unreachable states, not recoverable
/// errors. The codegen only receives validated input from the driver.
pub fn compile(program: &HirProgram) -> Vec<u8> {
    let mut compiler = Compiler::new();
    compiler.declare_all_runtime_funcs();

    // First pass: declare all top-level functions for forward references.
    for stmt in program {
        if let HirStmt::FuncDecl {
            name,
            params,
            ret_type,
            ..
        } = stmt
        {
            compiler.declare_tok_func(name, params, ret_type);
        }
    }

    // Second pass: compile all functions.
    let mut main_stmts = Vec::new();
    for stmt in program {
        match stmt {
            HirStmt::FuncDecl {
                name,
                params,
                ret_type,
                body,
            } => {
                // Store body for potential inlining at call sites
                compiler.func_bodies.insert(
                    name.clone(),
                    (params.clone(), ret_type.clone(), body.clone()),
                );
                compile_function(&mut compiler, name, params, ret_type, body);
            }
            other => {
                main_stmts.push(other.clone());
            }
        }
    }

    // Compile the main function (top-level statements).
    compile_main(&mut compiler, &main_stmts);

    // Third pass: compile pending lambdas (may recursively add more).
    while !compiler.pending_lambdas.is_empty() {
        let pending = std::mem::take(&mut compiler.pending_lambdas);
        for lambda in pending {
            if lambda.specialized_param_types.is_some() {
                compile_specialized_lambda_body(&mut compiler, &lambda);
            } else {
                compile_lambda_body(&mut compiler, &lambda);
            }
        }
    }

    // Emit the C entry point that calls _tok_main.
    compile_entry(&mut compiler);

    // Produce the object file bytes.
    let product: ObjectProduct = compiler.module.finish();
    product.emit().expect("codegen: failed to emit object file")
}

mod func;
use func::{
    compile_entry, compile_function, compile_lambda_body, compile_main,
    compile_specialized_lambda_body, load_captures_from_env,
};

// ─── FuncCtx helpers ──────────────────────────────────────────────────

impl<'a> FuncCtx<'a> {
    fn new_var(&mut self, ty: types::Type) -> Variable {
        let var = Variable::new(self.next_var);
        self.next_var += 1;
        self.builder.declare_var(var, ty);
        var
    }

    /// Look up or lazily declare a runtime function reference in the current function.
    fn get_runtime_func_ref(&mut self, name: &str) -> cranelift_codegen::ir::FuncRef {
        let func_id = *self
            .compiler
            .runtime_funcs
            .get(name)
            .unwrap_or_else(|| panic!("runtime function '{}' not declared", name));
        self.compiler
            .module
            .declare_func_in_func(func_id, self.builder.func)
    }

    /// Look up a declared Tok function reference.
    fn get_tok_func_ref(&mut self, name: &str) -> cranelift_codegen::ir::FuncRef {
        let func_id = *self
            .compiler
            .declared_funcs
            .get(name)
            .unwrap_or_else(|| panic!("tok function '{}' not declared", name));
        self.compiler
            .module
            .declare_func_in_func(func_id, self.builder.func)
    }

    /// Get a reference to a data object (string literal).
    fn get_data_ref(
        &mut self,
        data_id: cranelift_module::DataId,
    ) -> cranelift_codegen::ir::GlobalValue {
        self.compiler
            .module
            .declare_data_in_func(data_id, self.builder.func)
    }
}

mod stmt;
use stmt::{compile_body, compile_stmt};

mod expr;
pub(crate) use expr::compile_expr;

mod closures;
use closures::{compile_go_expr, compile_lambda_expr, compile_receive_expr, compile_select_expr};

mod binop;
use binop::compile_binop;

mod unary;
use unary::compile_unaryop;

mod calls;
use calls::compile_call;

mod inline;
use inline::{
    can_inline_closure_call, can_inline_hof, compile_inline_closure_call, compile_inline_filter,
    compile_inline_reduce, compile_print_call, contains_self_call, is_self_tail_recursive,
};

mod control;
use control::{compile_if, compile_loop};

mod coerce;
use coerce::{
    alloc_tokvalue_on_stack, coerce_if_branch, coerce_value, compile_expr_as_ptr, from_tokvalue,
    from_tokvalue_raw_data, to_bool, to_tokvalue, unwrap_any_ptr, zero_value, TAG_ARRAY, TAG_FLOAT,
    TAG_INT, TAG_STRING,
};

mod retype;
use retype::{retype_body, unwrap_return_stmts};

mod free_vars;
