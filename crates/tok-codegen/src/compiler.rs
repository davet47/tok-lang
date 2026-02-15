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
use cranelift_codegen::ir::{
    AbiParam, Block, Function, InstBuilder, MemFlags, StackSlotData, StackSlotKind, UserFuncName, Value,
};
use cranelift_codegen::isa;
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule, ObjectProduct};
use target_lexicon::Triple;

use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use tok_hir::hir::*;
use tok_types::Type;

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
#[allow(dead_code)]
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
    )
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
}

impl Compiler {
    fn new() -> Self {
        let mut settings_builder = settings::builder();
        settings_builder.set("opt_level", "speed").unwrap();
        settings_builder.set("is_pic", "true").unwrap();
        // Use the host triple
        let triple = Triple::from_str(&target_lexicon::HOST.to_string()).unwrap();
        let flags = settings::Flags::new(settings_builder);
        let isa = isa::lookup(triple.clone())
            .unwrap()
            .finish(flags)
            .unwrap();

        let call_conv = isa.default_call_conv();

        let obj_builder =
            ObjectBuilder::new(isa, "tok_output", cranelift_module::default_libcall_names())
                .unwrap();
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
            .unwrap();
        self.runtime_funcs.insert(name.to_string(), id);
        id
    }

    /// Declare all runtime functions we'll need.
    fn declare_all_runtime_funcs(&mut self) {
        // Print
        self.declare_runtime_func("tok_println_int", &[types::I64], &[]);
        self.declare_runtime_func("tok_println_float", &[types::F64], &[]);
        self.declare_runtime_func("tok_println_string", &[PTR], &[]);
        self.declare_runtime_func("tok_println_bool", &[types::I8], &[]);
        self.declare_runtime_func("tok_print_int", &[types::I64], &[]);
        self.declare_runtime_func("tok_print_float", &[types::F64], &[]);
        self.declare_runtime_func("tok_print_string", &[PTR], &[]);
        self.declare_runtime_func("tok_print_bool", &[types::I8], &[]);
        self.declare_runtime_func("tok_println", &[PTR, types::I64], &[]); // TokValue as 2 words
        self.declare_runtime_func("tok_print", &[PTR, types::I64], &[]);   // TokValue as 2 words

        // String
        self.declare_runtime_func("tok_string_alloc", &[PTR, types::I64], &[PTR]);
        self.declare_runtime_func("tok_string_concat", &[PTR, PTR], &[PTR]);
        self.declare_runtime_func("tok_string_len", &[PTR], &[types::I64]);
        self.declare_runtime_func("tok_string_eq", &[PTR, PTR], &[types::I8]);
        self.declare_runtime_func("tok_string_cmp", &[PTR, PTR], &[types::I64]);
        self.declare_runtime_func("tok_string_index", &[PTR, types::I64], &[PTR]);
        self.declare_runtime_func("tok_string_slice", &[PTR, types::I64, types::I64], &[PTR]);
        self.declare_runtime_func("tok_string_split", &[PTR, PTR], &[PTR]);
        self.declare_runtime_func("tok_string_trim", &[PTR], &[PTR]);
        self.declare_runtime_func("tok_int_to_string", &[types::I64], &[PTR]);
        self.declare_runtime_func("tok_float_to_string", &[types::F64], &[PTR]);
        self.declare_runtime_func("tok_bool_to_string", &[types::I8], &[PTR]);
        self.declare_runtime_func("tok_value_to_string", &[PTR, types::I64], &[PTR]);

        // Array
        self.declare_runtime_func("tok_array_alloc", &[], &[PTR]);
        self.declare_runtime_func("tok_array_push", &[PTR, PTR, types::I64], &[PTR]); // arr, tag+data
        self.declare_runtime_func("tok_array_get", &[PTR, types::I64], &[PTR, types::I64]); // -> TokValue
        self.declare_runtime_func("tok_array_set", &[PTR, types::I64, PTR, types::I64], &[]);
        self.declare_runtime_func("tok_array_len", &[PTR], &[types::I64]);
        self.declare_runtime_func("tok_array_slice", &[PTR, types::I64, types::I64], &[PTR]);
        self.declare_runtime_func("tok_array_sort", &[PTR], &[PTR]);
        self.declare_runtime_func("tok_array_rev", &[PTR], &[PTR]);
        self.declare_runtime_func("tok_array_flat", &[PTR], &[PTR]);
        self.declare_runtime_func("tok_array_uniq", &[PTR], &[PTR]);
        self.declare_runtime_func("tok_array_concat", &[PTR, PTR], &[PTR]);
        self.declare_runtime_func("tok_array_join", &[PTR, PTR], &[PTR]);
        self.declare_runtime_func("tok_array_filter", &[PTR, PTR], &[PTR]);
        self.declare_runtime_func("tok_array_reduce", &[PTR, types::I64, types::I64, PTR], &[types::I64, types::I64]);
        self.declare_runtime_func("tok_array_min", &[PTR], &[PTR, types::I64]); // -> TokValue
        self.declare_runtime_func("tok_array_max", &[PTR], &[PTR, types::I64]);
        self.declare_runtime_func("tok_array_sum", &[PTR], &[PTR, types::I64]);
        self.declare_runtime_func("tok_pmap", &[PTR, PTR], &[PTR]);

        // Map
        self.declare_runtime_func("tok_map_alloc", &[], &[PTR]);
        self.declare_runtime_func("tok_map_get", &[PTR, PTR], &[PTR, types::I64]); // -> TokValue
        self.declare_runtime_func("tok_map_set", &[PTR, PTR, PTR, types::I64], &[]);
        self.declare_runtime_func("tok_map_has", &[PTR, PTR], &[types::I8]);
        self.declare_runtime_func("tok_map_del", &[PTR, PTR], &[PTR]);
        self.declare_runtime_func("tok_map_keys", &[PTR], &[PTR]);
        self.declare_runtime_func("tok_map_vals", &[PTR], &[PTR]);
        self.declare_runtime_func("tok_map_len", &[PTR], &[types::I64]);

        // Tuple
        self.declare_runtime_func("tok_tuple_alloc", &[types::I64], &[PTR]);
        self.declare_runtime_func("tok_tuple_get", &[PTR, types::I64], &[PTR, types::I64]);
        self.declare_runtime_func("tok_tuple_set", &[PTR, types::I64, PTR, types::I64], &[]);
        self.declare_runtime_func("tok_tuple_len", &[PTR], &[types::I64]);

        // Value (Any-typed) helpers
        self.declare_runtime_func("tok_value_len", &[PTR, types::I64], &[types::I64]);
        self.declare_runtime_func("tok_value_index", &[PTR, types::I64, types::I64], &[PTR, types::I64]);

        // Closure
        self.declare_runtime_func("tok_closure_alloc", &[PTR, PTR, types::I32], &[PTR]);
        self.declare_runtime_func("tok_closure_get_fn", &[PTR], &[PTR]);
        self.declare_runtime_func("tok_closure_get_env", &[PTR], &[PTR]);
        // Environment allocation for captures: (count: I64) -> PTR
        self.declare_runtime_func("tok_env_alloc", &[types::I64], &[PTR]);

        // Channel
        self.declare_runtime_func("tok_channel_alloc", &[types::I64], &[PTR]);
        self.declare_runtime_func("tok_channel_send", &[PTR, PTR, types::I64], &[]);
        self.declare_runtime_func("tok_channel_recv", &[PTR], &[PTR, types::I64]);
        self.declare_runtime_func("tok_channel_try_send", &[PTR, PTR, types::I64], &[types::I8]);
        self.declare_runtime_func("tok_channel_try_recv", &[PTR, PTR], &[types::I8]);

        // Goroutine
        self.declare_runtime_func("tok_go", &[PTR, PTR], &[PTR]);
        self.declare_runtime_func("tok_handle_join", &[PTR], &[PTR, types::I64]);

        // Refcount
        self.declare_runtime_func("tok_rc_inc", &[PTR], &[]);
        self.declare_runtime_func("tok_rc_dec", &[PTR], &[types::I8]);

        // Conversion
        self.declare_runtime_func("tok_to_int", &[PTR, types::I64], &[types::I64]);
        self.declare_runtime_func("tok_to_float", &[PTR, types::I64], &[types::F64]);
        self.declare_runtime_func("tok_type_of", &[PTR, types::I64], &[PTR]);

        // Math
        self.declare_runtime_func("tok_abs_int", &[types::I64], &[types::I64]);
        self.declare_runtime_func("tok_abs_float", &[types::F64], &[types::F64]);
        self.declare_runtime_func("tok_value_abs", &[PTR, types::I64], &[PTR, types::I64]);
        self.declare_runtime_func("tok_floor", &[types::F64], &[types::I64]);
        self.declare_runtime_func("tok_value_floor", &[PTR, types::I64], &[PTR, types::I64]);
        self.declare_runtime_func("tok_ceil", &[types::F64], &[types::I64]);
        self.declare_runtime_func("tok_value_ceil", &[PTR, types::I64], &[PTR, types::I64]);
        self.declare_runtime_func("tok_rand", &[], &[types::F64]);

        // TokValue → concrete type extraction
        self.declare_runtime_func("tok_value_to_int", &[PTR, types::I64], &[types::I64]);
        self.declare_runtime_func("tok_value_to_float", &[PTR, types::I64], &[types::F64]);
        self.declare_runtime_func("tok_value_to_bool", &[PTR, types::I64], &[types::I8]);

        // Value ops (for Any type dispatch)
        self.declare_runtime_func("tok_value_add", &[PTR, types::I64, PTR, types::I64], &[PTR, types::I64]);
        self.declare_runtime_func("tok_value_sub", &[PTR, types::I64, PTR, types::I64], &[PTR, types::I64]);
        self.declare_runtime_func("tok_value_mul", &[PTR, types::I64, PTR, types::I64], &[PTR, types::I64]);
        self.declare_runtime_func("tok_value_div", &[PTR, types::I64, PTR, types::I64], &[PTR, types::I64]);
        self.declare_runtime_func("tok_value_mod", &[PTR, types::I64, PTR, types::I64], &[PTR, types::I64]);
        self.declare_runtime_func("tok_value_negate", &[PTR, types::I64], &[PTR, types::I64]);
        self.declare_runtime_func("tok_value_eq", &[PTR, types::I64, PTR, types::I64], &[types::I8]);
        self.declare_runtime_func("tok_value_lt", &[PTR, types::I64, PTR, types::I64], &[types::I8]);
        self.declare_runtime_func("tok_value_truthiness", &[PTR, types::I64], &[types::I8]);
        self.declare_runtime_func("tok_value_not", &[PTR, types::I64], &[types::I8]);

        // Utility
        self.declare_runtime_func("tok_clock", &[], &[types::I64]);
        self.declare_runtime_func("tok_exit", &[types::I64], &[]);
    }

    /// Declare a string literal as a data object, returning a DataId.
    fn declare_string_data(&mut self, s: &str) -> (cranelift_module::DataId, usize) {
        let name = format!("__tok_str_{}", self.string_literals.len());
        let data_id = self
            .module
            .declare_data(&name, Linkage::Local, false, false)
            .unwrap();
        let mut desc = DataDescription::new();
        desc.define(s.as_bytes().to_vec().into_boxed_slice());
        self.module.define_data(data_id, &desc).unwrap();
        let entry = (data_id, s.len());
        self.string_literals.push(entry);
        entry
    }

    /// Declare a Tok-level function (for forward references, recursion).
    fn declare_tok_func(
        &mut self,
        name: &str,
        params: &[HirParam],
        ret_type: &Type,
    ) -> FuncId {
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
            .unwrap();
        self.declared_funcs.insert(name.to_string(), id);
        let param_types: Vec<Type> = params.iter().map(|p| p.ty.clone()).collect();
        self.func_sigs.insert(name.to_string(), (param_types, ret_type.clone()));
        id
    }
}

// ─── Function-level codegen context ───────────────────────────────────

/// Per-function compilation state.
struct FuncCtx<'a> {
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
    ret_type: Type,
    /// Closures assigned to local variables where we know the FuncId at compile time.
    known_closures: HashMap<String, KnownClosure>,
    /// Set by Lambda compilation so the enclosing Assign can record it in known_closures.
    last_lambda_info: Option<(FuncId, Value, usize)>, // (func_id, env_ptr, pending_idx)
}

// We need a separate lifetime for the FunctionBuilderContext because
// the builder borrows it. The Compiler is borrowed mutably through
// a raw pointer trick -- we ensure safety by never aliasing.

/// Compile an HIR program to a native object file (bytes).
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
    product.emit().unwrap()
}

/// Compile a named function.
fn compile_function(
    compiler: &mut Compiler,
    name: &str,
    params: &[HirParam],
    ret_type: &Type,
    body: &[HirStmt],
) {
    let func_id = compiler.declare_tok_func(name, params, ret_type);
    let sig = compiler.module.declarations().get_function_decl(func_id).signature.clone();

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, func_id.as_u32());

    let mut func_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_builder_ctx);

    let entry_block = builder.create_block();
    let return_block = builder.create_block();

    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Set up return
    let is_any_return = matches!(ret_type, Type::Any);
    let return_var = if is_any_return {
        // Any return: return_block takes two params (tag, data)
        builder.append_block_param(return_block, types::I64);
        builder.append_block_param(return_block, types::I64);
        // Use a dummy variable index — won't be used directly
        let rv = Variable::new(0);
        builder.declare_var(rv, types::I64);
        let zero = zero_value(&mut builder, types::I64);
        builder.def_var(rv, zero);
        Some(rv)
    } else if let Some(ct) = cl_type(ret_type) {
        let rv = Variable::new(0);
        builder.declare_var(rv, ct);
        // Initialize return var to zero
        let zero = zero_value(&mut builder, ct);
        builder.def_var(rv, zero);
        // Add return block param
        builder.append_block_param(return_block, ct);
        Some(rv)
    } else {
        None
    };

    let next_var_start = if return_var.is_some() { 1 } else { 0 };

    // Build FuncCtx — we use a raw pointer to compiler to avoid borrow issues.
    // Safety: we only use compiler through func_ctx, never aliasing.
    let compiler_ptr = compiler as *mut Compiler;
    let mut func_ctx = FuncCtx {
        compiler: unsafe { &mut *compiler_ptr },
        builder,
        vars: HashMap::new(),
        next_var: next_var_start,
        return_block,
        return_var,
        loop_stack: Vec::new(),
        block_terminated: false,
        is_any_return,
        ret_type: ret_type.clone(),
        known_closures: HashMap::new(),
        last_lambda_info: None,
    };

    // Define parameters as variables
    // For Any params, each takes two block params (tag, data); we store as stack TokValue.
    let mut block_param_idx = 0;
    for (_i, param) in params.iter().enumerate() {
        if matches!(param.ty, Type::Any) {
            // Any param: two block params (tag, data), store as stack TokValue
            let tag_val = func_ctx.builder.block_params(entry_block)[block_param_idx];
            let data_val = func_ctx.builder.block_params(entry_block)[block_param_idx + 1];
            block_param_idx += 2;
            // Create stack slot and store
            let addr = alloc_tokvalue_on_stack(&mut func_ctx, tag_val, data_val);
            let var = func_ctx.new_var(PTR);
            func_ctx.builder.def_var(var, addr);
            func_ctx.vars.insert(param.name.clone(), (var, param.ty.clone()));
        } else if let Some(ct) = cl_type(&param.ty) {
            let var = func_ctx.new_var(ct);
            let param_val = func_ctx.builder.block_params(entry_block)[block_param_idx];
            block_param_idx += 1;
            func_ctx.builder.def_var(var, param_val);
            func_ctx.vars.insert(param.name.clone(), (var, param.ty.clone()));
        }
    }

    // Compile body
    let last_val = compile_body(&mut func_ctx, body, ret_type);

    // Jump to return block with value, but only if the current block isn't
    // already terminated (e.g., by a Return statement that already jumped).
    if !func_ctx.block_terminated {
        if is_any_return {
            if let Some(val) = last_val {
                // Determine the actual type of the last expression from the HIR,
                // so we use the correct type for to_tokvalue (not just ret_type=Any).
                let last_expr_ty = body.last()
                    .and_then(|s| match s {
                        HirStmt::Expr(e) => Some(e.ty.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| ret_type.clone());
                let (tag, data) = to_tokvalue(&mut func_ctx, val, &last_expr_ty);
                func_ctx.builder.ins().jump(return_block, &[tag, data]);
            } else {
                let zero = func_ctx.builder.ins().iconst(types::I64, 0);
                func_ctx.builder.ins().jump(return_block, &[zero, zero]);
            }
        } else if let Some(_rv) = return_var {
            if let Some(val) = last_val {
                func_ctx.builder.ins().jump(return_block, &[val]);
            } else {
                let default = zero_value(&mut func_ctx.builder, cl_type_or_i64(ret_type));
                func_ctx.builder.ins().jump(return_block, &[default]);
            }
        } else {
            func_ctx.builder.ins().jump(return_block, &[]);
        }
    }

    // Return block
    func_ctx.builder.switch_to_block(return_block);
    func_ctx.builder.seal_block(return_block);
    if is_any_return {
        let tag_ret = func_ctx.builder.block_params(return_block)[0];
        let data_ret = func_ctx.builder.block_params(return_block)[1];
        func_ctx.builder.ins().return_(&[tag_ret, data_ret]);
    } else if let Some(_rv) = return_var {
        let ret_val = func_ctx.builder.block_params(return_block)[0];
        func_ctx.builder.ins().return_(&[ret_val]);
    } else {
        func_ctx.builder.ins().return_(&[]);
    }

    func_ctx.builder.finalize();

    let mut ctx = Context::for_function(func);
    compiler
        .module
        .define_function(func_id, &mut ctx)
        .unwrap();
}

/// Compile a deferred lambda body into its own Cranelift function.
///
/// Lambda calling convention: (env_ptr: PTR, arg0_tag: I64, arg0_data: I64, ...) -> (I64, I64)
/// All params are passed as TokValue (tag, data) pairs, return is a TokValue pair.
fn compile_lambda_body(compiler: &mut Compiler, lambda: &PendingLambda) {
    let sig = compiler.module.declarations().get_function_decl(lambda.func_id).signature.clone();

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, lambda.func_id.as_u32());

    let mut func_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_builder_ctx);

    let entry_block = builder.create_block();
    let return_block = builder.create_block();

    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Return block takes (tag: I64, data: I64)
    builder.append_block_param(return_block, types::I64);
    builder.append_block_param(return_block, types::I64);

    // Use a dummy return variable (won't be used directly since we use block params)
    let rv = Variable::new(0);
    builder.declare_var(rv, types::I64);
    let zero_val = builder.ins().iconst(types::I64, 0);
    builder.def_var(rv, zero_val);

    let compiler_ptr = compiler as *mut Compiler;
    let mut func_ctx = FuncCtx {
        compiler: unsafe { &mut *compiler_ptr },
        builder,
        vars: HashMap::new(),
        next_var: 1, // 0 is used by return var
        return_block,
        return_var: Some(rv),
        loop_stack: Vec::new(),
        block_terminated: false,
        is_any_return: true, // lambdas always return (tag, data)
        ret_type: lambda.ret_type.clone(),
        known_closures: HashMap::new(),
        last_lambda_info: None,
    };

    // Bind parameters: first block param is env_ptr, then (tag, data) pairs
    let env_ptr_val = func_ctx.builder.block_params(entry_block)[0];
    let mut block_param_idx = 1; // skip env_ptr at index 0
    for param in &lambda.params {
        let tag_val = func_ctx.builder.block_params(entry_block)[block_param_idx];
        let data_val = func_ctx.builder.block_params(entry_block)[block_param_idx + 1];
        block_param_idx += 2;
        // Store as stack-allocated TokValue and bind to param name as Any
        let addr = alloc_tokvalue_on_stack(&mut func_ctx, tag_val, data_val);
        let var = func_ctx.new_var(PTR);
        func_ctx.builder.def_var(var, addr);
        func_ctx.vars.insert(param.name.clone(), (var, Type::Any));
    }

    // Load captured variables from env_ptr
    for (i, cap) in lambda.captures.iter().enumerate() {
        let offset = (i * 16) as i32;
        let tag = func_ctx.builder.ins().load(types::I64, MemFlags::trusted(), env_ptr_val, offset);
        let data = func_ctx.builder.ins().load(types::I64, MemFlags::trusted(), env_ptr_val, offset + 8);
        // Store as stack-allocated TokValue and bind to captured var name as Any
        let addr = alloc_tokvalue_on_stack(&mut func_ctx, tag, data);
        let var = func_ctx.new_var(PTR);
        func_ctx.builder.def_var(var, addr);
        func_ctx.vars.insert(cap.name.clone(), (var, Type::Any));
    }

    // Compile body
    let last_val = compile_body(&mut func_ctx, &lambda.body, &lambda.ret_type);

    // Jump to return block with value
    if !func_ctx.block_terminated {
        if let Some(val) = last_val {
            // Use the actual type of the last expression, not just ret_type
            let last_expr_ty = lambda.body.last()
                .and_then(|s| match s {
                    HirStmt::Expr(e) => Some(e.ty.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| lambda.ret_type.clone());
            let (tag, data) = to_tokvalue(&mut func_ctx, val, &last_expr_ty);
            func_ctx.builder.ins().jump(return_block, &[tag, data]);
        } else {
            let zero = func_ctx.builder.ins().iconst(types::I64, 0);
            func_ctx.builder.ins().jump(return_block, &[zero, zero]);
        }
    }

    // Return block
    func_ctx.builder.switch_to_block(return_block);
    func_ctx.builder.seal_block(return_block);
    let tag_ret = func_ctx.builder.block_params(return_block)[0];
    let data_ret = func_ctx.builder.block_params(return_block)[1];
    func_ctx.builder.ins().return_(&[tag_ret, data_ret]);

    func_ctx.builder.finalize();

    let mut ctx = Context::for_function(func);
    compiler
        .module
        .define_function(lambda.func_id, &mut ctx)
        .unwrap();
}

/// Compile a specialized lambda body with native-typed calling convention.
///
/// Specialized calling convention: (env_ptr: PTR, arg0: T0, arg1: T1, ...) -> RetT
/// Params are native types, no boxing/unboxing.
fn compile_specialized_lambda_body(compiler: &mut Compiler, lambda: &PendingLambda) {
    let spec_types = lambda.specialized_param_types.as_ref().unwrap();
    let sig = compiler.module.declarations().get_function_decl(lambda.func_id).signature.clone();

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, lambda.func_id.as_u32());

    let mut func_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_builder_ctx);

    let entry_block = builder.create_block();
    let return_block = builder.create_block();

    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Return block takes a single native-typed value
    let cl_ret = cl_type_or_i64(&lambda.ret_type);
    builder.append_block_param(return_block, cl_ret);

    let rv = Variable::new(0);
    builder.declare_var(rv, cl_ret);
    let zero_val = builder.ins().iconst(types::I64, 0);
    builder.def_var(rv, zero_val);

    let compiler_ptr = compiler as *mut Compiler;
    let mut func_ctx = FuncCtx {
        compiler: unsafe { &mut *compiler_ptr },
        builder,
        vars: HashMap::new(),
        next_var: 1,
        return_block,
        return_var: Some(rv),
        loop_stack: Vec::new(),
        block_terminated: false,
        is_any_return: false, // specialized: single native return
        ret_type: lambda.ret_type.clone(),
        known_closures: HashMap::new(),
        last_lambda_info: None,
    };

    // Bind parameters: first block param is env_ptr, then one native value per param
    let env_ptr_val = func_ctx.builder.block_params(entry_block)[0];
    for (i, (param, param_ty)) in lambda.params.iter().zip(spec_types.iter()).enumerate() {
        let val = func_ctx.builder.block_params(entry_block)[1 + i];
        let ct = cl_type_or_i64(param_ty);
        let var = func_ctx.new_var(ct);
        func_ctx.builder.def_var(var, val);
        func_ctx.vars.insert(param.name.clone(), (var, param_ty.clone()));
    }

    // Load captured variables from env_ptr
    for (i, cap) in lambda.captures.iter().enumerate() {
        let offset = (i * 16) as i32;
        match &cap.ty {
            Type::Int => {
                // Extract data field directly as i64 (skip tag)
                let data = func_ctx.builder.ins().load(types::I64, MemFlags::trusted(), env_ptr_val, offset + 8);
                let var = func_ctx.new_var(types::I64);
                func_ctx.builder.def_var(var, data);
                func_ctx.vars.insert(cap.name.clone(), (var, Type::Int));
            }
            Type::Float => {
                let data = func_ctx.builder.ins().load(types::I64, MemFlags::trusted(), env_ptr_val, offset + 8);
                let fval = func_ctx.builder.ins().bitcast(types::F64, MemFlags::new(), data);
                let var = func_ctx.new_var(types::F64);
                func_ctx.builder.def_var(var, fval);
                func_ctx.vars.insert(cap.name.clone(), (var, Type::Float));
            }
            Type::Bool => {
                let data = func_ctx.builder.ins().load(types::I64, MemFlags::trusted(), env_ptr_val, offset + 8);
                let bval = func_ctx.builder.ins().ireduce(types::I8, data);
                let var = func_ctx.new_var(types::I8);
                func_ctx.builder.def_var(var, bval);
                func_ctx.vars.insert(cap.name.clone(), (var, Type::Bool));
            }
            _ => {
                // Unknown type — load as Any (TokValue on stack)
                let tag = func_ctx.builder.ins().load(types::I64, MemFlags::trusted(), env_ptr_val, offset);
                let data = func_ctx.builder.ins().load(types::I64, MemFlags::trusted(), env_ptr_val, offset + 8);
                let addr = alloc_tokvalue_on_stack(&mut func_ctx, tag, data);
                let var = func_ctx.new_var(PTR);
                func_ctx.builder.def_var(var, addr);
                func_ctx.vars.insert(cap.name.clone(), (var, Type::Any));
            }
        }
    }

    // Compile body
    let last_val = compile_body(&mut func_ctx, &lambda.body, &lambda.ret_type);

    // Jump to return block with native value
    if !func_ctx.block_terminated {
        if let Some(val) = last_val {
            let last_expr_ty = lambda.body.last()
                .and_then(|s| match s {
                    HirStmt::Expr(e) => Some(e.ty.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| lambda.ret_type.clone());
            let coerced = coerce_value(&mut func_ctx, val, &last_expr_ty, &lambda.ret_type);
            func_ctx.builder.ins().jump(return_block, &[coerced]);
        } else {
            let zero = func_ctx.builder.ins().iconst(cl_ret, 0);
            func_ctx.builder.ins().jump(return_block, &[zero]);
        }
    }

    // Return block
    func_ctx.builder.switch_to_block(return_block);
    func_ctx.builder.seal_block(return_block);
    let ret_val = func_ctx.builder.block_params(return_block)[0];
    func_ctx.builder.ins().return_(&[ret_val]);

    func_ctx.builder.finalize();

    let mut ctx = Context::for_function(func);
    compiler
        .module
        .define_function(lambda.func_id, &mut ctx)
        .unwrap();
}

/// Compile top-level statements into `_tok_main`.
fn compile_main(compiler: &mut Compiler, stmts: &[HirStmt]) {
    let sig = compiler.module.make_signature();
    // _tok_main returns void
    let func_id = compiler
        .module
        .declare_function("_tok_main", Linkage::Export, &sig)
        .unwrap();

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, func_id.as_u32());

    let mut func_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_builder_ctx);

    let entry_block = builder.create_block();
    let return_block = builder.create_block();

    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let compiler_ptr = compiler as *mut Compiler;
    let mut func_ctx = FuncCtx {
        compiler: unsafe { &mut *compiler_ptr },
        builder,
        vars: HashMap::new(),
        next_var: 0,
        return_block,
        return_var: None,
        loop_stack: Vec::new(),
        block_terminated: false,
        is_any_return: false,
        ret_type: Type::Nil,
        known_closures: HashMap::new(),
        last_lambda_info: None,
    };

    compile_body(&mut func_ctx, stmts, &Type::Nil);

    if !func_ctx.block_terminated {
        func_ctx.builder.ins().jump(return_block, &[]);
    }
    func_ctx.builder.switch_to_block(return_block);
    func_ctx.builder.seal_block(return_block);
    func_ctx.builder.ins().return_(&[]);
    func_ctx.builder.finalize();

    let mut ctx = Context::for_function(func);
    if std::env::var("CLIF_DUMP").is_ok() {
        eprintln!("=== _tok_main IR ===\n{}", ctx.func.display());
    }
    compiler
        .module
        .define_function(func_id, &mut ctx)
        .unwrap();
}

/// Compile the C `main` entry point that calls `_tok_main`.
fn compile_entry(compiler: &mut Compiler) {
    let mut sig = compiler.module.make_signature();
    sig.params.push(AbiParam::new(types::I32)); // argc
    sig.params.push(AbiParam::new(PTR));         // argv
    sig.returns.push(AbiParam::new(types::I32)); // exit code

    let func_id = compiler
        .module
        .declare_function("main", Linkage::Export, &sig)
        .unwrap();

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, func_id.as_u32());

    // Declare _tok_main reference before creating builder (avoids borrow conflict)
    let tok_main_sig = compiler.module.make_signature();
    let tok_main_id = compiler
        .module
        .declare_function("_tok_main", Linkage::Export, &tok_main_sig)
        .unwrap();
    let tok_main_ref = compiler.module.declare_func_in_func(tok_main_id, &mut func);

    let mut func_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_builder_ctx);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Call _tok_main
    builder.ins().call(tok_main_ref, &[]);

    // Return 0
    let zero = builder.ins().iconst(types::I32, 0);
    builder.ins().return_(&[zero]);

    builder.finalize();

    let mut ctx = Context::for_function(func);
    compiler
        .module
        .define_function(func_id, &mut ctx)
        .unwrap();
}

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

// ─── Body / statement compilation ─────────────────────────────────────

/// Compile a sequence of statements, returning the value of the last expression.
fn compile_body(
    ctx: &mut FuncCtx,
    stmts: &[HirStmt],
    _expected_type: &Type,
) -> Option<Value> {
    let mut last_val = None;
    for stmt in stmts {
        last_val = compile_stmt(ctx, stmt);
    }
    last_val
}

/// Compile a single HIR statement. Returns a value if the statement is an expression.
fn compile_stmt(ctx: &mut FuncCtx, stmt: &HirStmt) -> Option<Value> {
    match stmt {
        HirStmt::Assign { name, ty, value } => {
            ctx.last_lambda_info = None;
            let val = compile_expr(ctx, value);
            // If the RHS was a lambda, record it for direct-call optimization
            if let Some((func_id, env_ptr, pending_idx)) = ctx.last_lambda_info.take() {
                ctx.known_closures.insert(name.clone(), KnownClosure {
                    func_id,
                    env_ptr,
                    pending_idx,
                    specialized: None,
                });
            } else {
                // Variable reassigned to non-lambda — invalidate
                ctx.known_closures.remove(name.as_str());
            }
            if let Some(v) = val {
                if let Some((var, existing_ty)) = ctx.vars.get(name).cloned() {
                    // Coerce value to match the variable's existing type
                    let coerced = match (&existing_ty, &value.ty) {
                        // Value is Any but variable is concrete — extract via runtime
                        (Type::Int, Type::Any) => {
                            let func_ref = ctx.get_runtime_func_ref("tok_value_to_int");
                            let (tag, data) = to_tokvalue(ctx, v, &Type::Any);
                            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                            ctx.builder.inst_results(call)[0]
                        }
                        (Type::Float, Type::Any) => {
                            let func_ref = ctx.get_runtime_func_ref("tok_value_to_float");
                            let (tag, data) = to_tokvalue(ctx, v, &Type::Any);
                            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                            ctx.builder.inst_results(call)[0]
                        }
                        (Type::Bool, Type::Any) => {
                            let func_ref = ctx.get_runtime_func_ref("tok_value_to_bool");
                            let (tag, data) = to_tokvalue(ctx, v, &Type::Any);
                            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                            ctx.builder.inst_results(call)[0]
                        }
                        (et, Type::Any) if !matches!(et, Type::Any) => {
                            // For pointer types (Str, Array, Map, etc.), extract data field
                            let (_, data) = to_tokvalue(ctx, v, &Type::Any);
                            data
                        }
                        // Variable is Any but value is concrete — wrap into TokValue
                        (Type::Any, vt) if !matches!(vt, Type::Any | Type::Nil | Type::Never) => {
                            let (tag, data) = to_tokvalue(ctx, v, &value.ty);
                            alloc_tokvalue_on_stack(ctx, tag, data)
                        }
                        _ => v,
                    };
                    ctx.builder.def_var(var, coerced);
                } else {
                    let ct = cl_type_or_i64(ty);
                    let var = ctx.new_var(ct);
                    ctx.builder.def_var(var, v);
                    ctx.vars.insert(name.clone(), (var, ty.clone()));
                }
            }
            None
        }

        HirStmt::FuncDecl { .. } => {
            // Nested function declarations — not yet supported in codegen.
            // They should have been lifted out during HIR lowering.
            None
        }

        HirStmt::IndexAssign {
            target,
            index,
            value,
        } => {
            let target_val = compile_expr(ctx, target).unwrap();
            let idx_val = compile_expr(ctx, index).unwrap();
            let val = compile_expr(ctx, value).unwrap();
            // Call appropriate runtime set function based on target type
            match &target.ty {
                Type::Array(_) => {
                    // Pack value as TokValue for array_set
                    let idx = unwrap_any_ptr(ctx, idx_val, &index.ty);
                    let (tag, data) = to_tokvalue(ctx, val, &value.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_set");
                    ctx.builder.ins().call(func_ref, &[target_val, idx, tag, data]);
                }
                Type::Map(_) => {
                    // Key must be a string pointer; unwrap from Any if needed
                    let key = unwrap_any_ptr(ctx, idx_val, &index.ty);
                    let (tag, data) = to_tokvalue(ctx, val, &value.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_map_set");
                    ctx.builder.ins().call(func_ref, &[target_val, key, tag, data]);
                }
                _ => {}
            }
            None
        }

        HirStmt::MemberAssign {
            target,
            field,
            value,
        } => {
            let target_val = compile_expr(ctx, target).unwrap();
            let val = compile_expr(ctx, value).unwrap();
            // Allocate key string
            let (data_id, len) = ctx.compiler.declare_string_data(field);
            let gv = ctx.get_data_ref(data_id);
            let key_ptr = ctx.builder.ins().global_value(PTR, gv);
            let key_len = ctx.builder.ins().iconst(types::I64, len as i64);
            let func_ref = ctx.get_runtime_func_ref("tok_string_alloc");
            let call = ctx.builder.ins().call(func_ref, &[key_ptr, key_len]);
            let key_str = ctx.builder.inst_results(call)[0];
            // Set
            let (tag, data) = to_tokvalue(ctx, val, &value.ty);
            let set_ref = ctx.get_runtime_func_ref("tok_map_set");
            ctx.builder.ins().call(set_ref, &[target_val, key_str, tag, data]);
            None
        }

        HirStmt::Expr(expr) => {
            compile_expr(ctx, expr)
        }

        HirStmt::Return(opt_expr) => {
            if let Some(expr) = opt_expr {
                let val = compile_expr(ctx, expr);
                if let Some(v) = val {
                    if ctx.is_any_return {
                        // Extract tag+data from TokValue pointer
                        let (tag, data) = to_tokvalue(ctx, v, &expr.ty);
                        ctx.builder.ins().jump(ctx.return_block, &[tag, data]);
                    } else {
                        ctx.builder.ins().jump(ctx.return_block, &[v]);
                    }
                } else {
                    if ctx.is_any_return {
                        let zero = ctx.builder.ins().iconst(types::I64, 0);
                        ctx.builder.ins().jump(ctx.return_block, &[zero, zero]);
                    } else {
                        ctx.builder.ins().jump(ctx.return_block, &[]);
                    }
                }
            } else {
                if ctx.is_any_return {
                    let zero = ctx.builder.ins().iconst(types::I64, 0);
                    ctx.builder.ins().jump(ctx.return_block, &[zero, zero]);
                } else {
                    ctx.builder.ins().jump(ctx.return_block, &[]);
                }
            }
            // Create a dead block with a trap for unreachable code after return
            let dead_block = ctx.builder.create_block();
            ctx.builder.switch_to_block(dead_block);
            ctx.builder.seal_block(dead_block);
            ctx.block_terminated = true;
            None
        }

        HirStmt::Break => {
            if let Some(&(_, break_block)) = ctx.loop_stack.last() {
                ctx.builder.ins().jump(break_block, &[]);
                ctx.block_terminated = true;
            }
            None
        }

        HirStmt::Continue => {
            if let Some(&(continue_block, _)) = ctx.loop_stack.last() {
                ctx.builder.ins().jump(continue_block, &[]);
                let dead_block = ctx.builder.create_block();
                ctx.builder.switch_to_block(dead_block);
                ctx.builder.seal_block(dead_block);
                ctx.block_terminated = true;
            }
            None
        }

        HirStmt::Import(_path) => {
            // Imports handled at a higher level (whole-program compilation).
            None
        }
    }
}

// ─── Expression compilation ───────────────────────────────────────────

/// Compile an HIR expression, returning the Cranelift Value.
/// Returns None for Nil-typed expressions.
fn compile_expr(ctx: &mut FuncCtx, expr: &HirExpr) -> Option<Value> {
    match &expr.kind {
        HirExprKind::Int(n) => {
            Some(ctx.builder.ins().iconst(types::I64, *n))
        }

        HirExprKind::Float(f) => {
            Some(ctx.builder.ins().f64const(*f))
        }

        HirExprKind::Bool(b) => {
            Some(ctx.builder.ins().iconst(types::I8, *b as i64))
        }

        HirExprKind::Nil => {
            // Return a zero i64 as a nil sentinel in contexts that need a value
            if cl_type(&expr.ty).is_some() {
                Some(ctx.builder.ins().iconst(types::I64, 0))
            } else {
                None
            }
        }

        HirExprKind::Str(s) => {
            let (data_id, len) = ctx.compiler.declare_string_data(s);
            let gv = ctx.get_data_ref(data_id);
            let ptr = ctx.builder.ins().global_value(PTR, gv);
            let len_val = ctx.builder.ins().iconst(types::I64, len as i64);
            let func_ref = ctx.get_runtime_func_ref("tok_string_alloc");
            let call = ctx.builder.ins().call(func_ref, &[ptr, len_val]);
            Some(ctx.builder.inst_results(call)[0])
        }

        HirExprKind::Ident(name) => {
            if let Some((var, var_ty)) = ctx.vars.get(name).cloned() {
                let raw = ctx.builder.use_var(var);
                // If the variable is stored as Any but the expression type is concrete,
                // coerce from Any → concrete. This happens with captured variables
                // in lambdas, whose vars are stored as Any (TokValue ptr) but the HIR
                // still has the original concrete type from type checking.
                if matches!(var_ty, Type::Any) && !matches!(&expr.ty, Type::Any | Type::Nil | Type::Never) {
                    Some(coerce_value(ctx, raw, &Type::Any, &expr.ty))
                } else if !matches!(var_ty, Type::Any) && matches!(&expr.ty, Type::Any) {
                    // Variable stored as concrete (e.g., Int) but HIR thinks it's Any
                    // (happens when variable was first assigned concrete, then reassigned
                    // from an Any-typed expression — codegen keeps the original type).
                    // Wrap the concrete value into a TokValue pointer.
                    Some(coerce_value(ctx, raw, &var_ty, &Type::Any))
                } else {
                    Some(raw)
                }
            } else if ctx.compiler.declared_funcs.contains_key(name.as_str()) {
                // Declared function used as a value — create a trampoline wrapper
                // with the closure calling convention, then wrap in a TokClosure.
                let (param_types, ret_type) = ctx.compiler.func_sigs.get(name.as_str()).unwrap().clone();
                let trampoline_name = format!("__tok_tramp_{}", name);

                // Create trampoline as a PendingLambda that just calls the function
                let tramp_params: Vec<HirParam> = param_types.iter().enumerate()
                    .map(|(i, ty)| HirParam { name: format!("__p{}", i), ty: ty.clone() })
                    .collect();
                let call_args: Vec<HirExpr> = tramp_params.iter()
                    .map(|p| HirExpr::new(HirExprKind::Ident(p.name.clone()), p.ty.clone()))
                    .collect();
                let call_expr = HirExpr::new(
                    HirExprKind::Call {
                        func: Box::new(HirExpr::new(HirExprKind::Ident(name.clone()), Type::Any)),
                        args: call_args,
                    },
                    ret_type.clone(),
                );

                // Declare trampoline function signature: (env, tag0, data0, ...) -> (tag, data)
                let mut sig = ctx.compiler.module.make_signature();
                sig.params.push(AbiParam::new(PTR)); // env_ptr
                for _ in &param_types {
                    sig.params.push(AbiParam::new(types::I64)); // tag
                    sig.params.push(AbiParam::new(types::I64)); // data
                }
                sig.returns.push(AbiParam::new(types::I64)); // result tag
                sig.returns.push(AbiParam::new(types::I64)); // result data

                let tramp_func_id = ctx.compiler.module
                    .declare_function(&trampoline_name, Linkage::Local, &sig)
                    .unwrap();

                ctx.compiler.pending_lambdas.push(PendingLambda {
                    name: trampoline_name,
                    func_id: tramp_func_id,
                    params: tramp_params,
                    ret_type: ret_type.clone(),
                    body: vec![HirStmt::Expr(call_expr)],
                    captures: vec![],
                    specialized_param_types: None,
                });

                let tramp_ref = ctx.compiler.module.declare_func_in_func(tramp_func_id, ctx.builder.func);
                let fn_ptr = ctx.builder.ins().func_addr(PTR, tramp_ref);
                let env_ptr = ctx.builder.ins().iconst(PTR, 0);
                let arity_val = ctx.builder.ins().iconst(types::I32, param_types.len() as i64);
                let alloc_ref = ctx.get_runtime_func_ref("tok_closure_alloc");
                let call = ctx.builder.ins().call(alloc_ref, &[fn_ptr, env_ptr, arity_val]);
                Some(ctx.builder.inst_results(call)[0])
            } else {
                // Unknown variable — return 0 as fallback
                Some(ctx.builder.ins().iconst(types::I64, 0))
            }
        }

        HirExprKind::Array(elems) => {
            // Allocate empty array, then push each element
            let alloc_ref = ctx.get_runtime_func_ref("tok_array_alloc");
            let call = ctx.builder.ins().call(alloc_ref, &[]);
            let mut arr = ctx.builder.inst_results(call)[0];

            for elem in elems {
                let val = compile_expr(ctx, elem).unwrap_or_else(|| {
                    ctx.builder.ins().iconst(types::I64, 0)
                });
                let (tag, data) = to_tokvalue(ctx, val, &elem.ty);
                let push_ref = ctx.get_runtime_func_ref("tok_array_push");
                let push_call = ctx.builder.ins().call(push_ref, &[arr, tag, data]);
                arr = ctx.builder.inst_results(push_call)[0];
            }
            Some(arr)
        }

        HirExprKind::Map(entries) => {
            let alloc_ref = ctx.get_runtime_func_ref("tok_map_alloc");
            let call = ctx.builder.ins().call(alloc_ref, &[]);
            let map = ctx.builder.inst_results(call)[0];

            for (key, val_expr) in entries {
                // Allocate key string
                let (data_id, len) = ctx.compiler.declare_string_data(key);
                let gv = ctx.get_data_ref(data_id);
                let key_ptr = ctx.builder.ins().global_value(PTR, gv);
                let key_len = ctx.builder.ins().iconst(types::I64, len as i64);
                let str_ref = ctx.get_runtime_func_ref("tok_string_alloc");
                let str_call = ctx.builder.ins().call(str_ref, &[key_ptr, key_len]);
                let key_str = ctx.builder.inst_results(str_call)[0];

                let val = compile_expr(ctx, val_expr).unwrap_or_else(|| {
                    ctx.builder.ins().iconst(types::I64, 0)
                });
                let (tag, data) = to_tokvalue(ctx, val, &val_expr.ty);
                let set_ref = ctx.get_runtime_func_ref("tok_map_set");
                ctx.builder.ins().call(set_ref, &[map, key_str, tag, data]);
            }
            Some(map)
        }

        HirExprKind::Tuple(elems) => {
            let count = ctx.builder.ins().iconst(types::I64, elems.len() as i64);
            let alloc_ref = ctx.get_runtime_func_ref("tok_tuple_alloc");
            let call = ctx.builder.ins().call(alloc_ref, &[count]);
            let tuple = ctx.builder.inst_results(call)[0];

            for (i, elem) in elems.iter().enumerate() {
                let val = compile_expr(ctx, elem).unwrap_or_else(|| {
                    ctx.builder.ins().iconst(types::I64, 0)
                });
                let idx = ctx.builder.ins().iconst(types::I64, i as i64);
                let (tag, data) = to_tokvalue(ctx, val, &elem.ty);
                let set_ref = ctx.get_runtime_func_ref("tok_tuple_set");
                ctx.builder.ins().call(set_ref, &[tuple, idx, tag, data]);
            }
            Some(tuple)
        }

        HirExprKind::BinOp { op, left, right } => {
            compile_binop(ctx, *op, left, right, &expr.ty)
        }

        HirExprKind::UnaryOp { op, operand } => {
            compile_unaryop(ctx, *op, operand, &expr.ty)
        }

        HirExprKind::Index { target, index } => {
            let target_val = compile_expr(ctx, target).unwrap();
            let idx_val = compile_expr(ctx, index).unwrap();
            match &target.ty {
                Type::Array(_) => {
                    let func_ref = ctx.get_runtime_func_ref("tok_array_get");
                    let call = ctx.builder.ins().call(func_ref, &[target_val, idx_val]);
                    let results = ctx.builder.inst_results(call);
                    // Returns TokValue (tag, data) — extract based on expected type
                    let tag = results[0];
                    let data = results[1];
                    Some(from_tokvalue(ctx, tag, data, &expr.ty))
                }
                Type::Str => {
                    let func_ref = ctx.get_runtime_func_ref("tok_string_index");
                    let call = ctx.builder.ins().call(func_ref, &[target_val, idx_val]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                Type::Tuple(_) => {
                    let func_ref = ctx.get_runtime_func_ref("tok_tuple_get");
                    let call = ctx.builder.ins().call(func_ref, &[target_val, idx_val]);
                    let results = ctx.builder.inst_results(call);
                    let tag = results[0];
                    let data = results[1];
                    Some(from_tokvalue(ctx, tag, data, &expr.ty))
                }
                Type::Map(_) => {
                    // Index by string key — unwrap key from Any if needed
                    let key = unwrap_any_ptr(ctx, idx_val, &index.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_map_get");
                    let call = ctx.builder.ins().call(func_ref, &[target_val, key]);
                    let results = ctx.builder.inst_results(call);
                    let tag = results[0];
                    let data = results[1];
                    Some(from_tokvalue(ctx, tag, data, &expr.ty))
                }
                Type::Any | Type::Optional(_) | Type::Result(_) => {
                    // Extract (tag, data) from TokValue pointer, call tok_value_index
                    let (t_tag, t_data) = to_tokvalue(ctx, target_val, &target.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_value_index");
                    let call = ctx.builder.ins().call(func_ref, &[t_tag, t_data, idx_val]);
                    let results = ctx.builder.inst_results(call);
                    Some(from_tokvalue(ctx, results[0], results[1], &expr.ty))
                }
                _ => Some(target_val),
            }
        }

        HirExprKind::Member { target, field } => {
            let target_val = compile_expr(ctx, target).unwrap();
            // Allocate field name as string, call map_get
            let (data_id, len) = ctx.compiler.declare_string_data(field);
            let gv = ctx.get_data_ref(data_id);
            let key_ptr = ctx.builder.ins().global_value(PTR, gv);
            let key_len = ctx.builder.ins().iconst(types::I64, len as i64);
            let str_ref = ctx.get_runtime_func_ref("tok_string_alloc");
            let str_call = ctx.builder.ins().call(str_ref, &[key_ptr, key_len]);
            let key_str = ctx.builder.inst_results(str_call)[0];

            match &target.ty {
                Type::Tuple(_) => {
                    // Numeric field access (.0, .1, etc.)
                    if let Ok(idx) = field.parse::<i64>() {
                        let idx_val = ctx.builder.ins().iconst(types::I64, idx);
                        let func_ref = ctx.get_runtime_func_ref("tok_tuple_get");
                        let call = ctx.builder.ins().call(func_ref, &[target_val, idx_val]);
                        let results = ctx.builder.inst_results(call);
                        Some(from_tokvalue(ctx, results[0], results[1], &expr.ty))
                    } else {
                        Some(target_val)
                    }
                }
                Type::Any | Type::Optional(_) | Type::Result(_) => {
                    // target_val is a PTR to stack TokValue — extract the map pointer
                    let map_ptr = ctx.builder.ins().load(types::I64, MemFlags::trusted(), target_val, 8);
                    let func_ref = ctx.get_runtime_func_ref("tok_map_get");
                    let call = ctx.builder.ins().call(func_ref, &[map_ptr, key_str]);
                    let results = ctx.builder.inst_results(call);
                    Some(from_tokvalue(ctx, results[0], results[1], &expr.ty))
                }
                _ => {
                    let func_ref = ctx.get_runtime_func_ref("tok_map_get");
                    let call = ctx.builder.ins().call(func_ref, &[target_val, key_str]);
                    let results = ctx.builder.inst_results(call);
                    Some(from_tokvalue(ctx, results[0], results[1], &expr.ty))
                }
            }
        }

        HirExprKind::Call { func, args } => {
            compile_call(ctx, func, args, &expr.ty)
        }

        HirExprKind::RuntimeCall { name, args } => {
            // Special-case filter/reduce: closure arg needs special handling
            match name.as_str() {
                "tok_array_filter" => {
                    // Inline filter when lambda is literal and element type is concrete
                    if can_inline_hof(&args[1], &args[0].ty, 1) {
                        return compile_inline_filter(ctx, &args[0], &args[1], &expr.ty);
                    }
                    // Fallback: runtime call
                    let arr_raw = compile_expr(ctx, &args[0]).unwrap();
                    let arr = if matches!(&args[0].ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                        ctx.builder.ins().load(types::I64, MemFlags::trusted(), arr_raw, 8)
                    } else {
                        arr_raw
                    };
                    let closure = compile_expr(ctx, &args[1]).unwrap();
                    let func_ref = ctx.get_runtime_func_ref("tok_array_filter");
                    let call = ctx.builder.ins().call(func_ref, &[arr, closure]);
                    let result = ctx.builder.inst_results(call)[0];
                    if matches!(&expr.ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                        let tag = ctx.builder.ins().iconst(types::I64, TAG_ARRAY as i64);
                        return Some(alloc_tokvalue_on_stack(ctx, tag, result));
                    }
                    return Some(result);
                }
                "tok_array_reduce" => {
                    // Inline reduce when lambda is literal and element type is concrete
                    if can_inline_hof(&args[2], &args[0].ty, 2) {
                        return compile_inline_reduce(ctx, &args[0], &args[1], &args[2], &expr.ty);
                    }
                    // Fallback: runtime call
                    let arr_raw = compile_expr(ctx, &args[0]).unwrap();
                    let arr = if matches!(&args[0].ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                        ctx.builder.ins().load(types::I64, MemFlags::trusted(), arr_raw, 8)
                    } else {
                        arr_raw
                    };
                    let init_val = compile_expr(ctx, &args[1]);
                    let closure = compile_expr(ctx, &args[2]).unwrap();
                    let (init_tag, init_data) = if let Some(iv) = init_val {
                        to_tokvalue(ctx, iv, &args[1].ty)
                    } else {
                        let zero = ctx.builder.ins().iconst(types::I64, 0);
                        (zero, zero)
                    };
                    let func_ref = ctx.get_runtime_func_ref("tok_array_reduce");
                    let call = ctx.builder.ins().call(func_ref, &[arr, init_tag, init_data, closure]);
                    let results = ctx.builder.inst_results(call);
                    return Some(from_tokvalue(ctx, results[0], results[1], &expr.ty));
                }
                "tok_array_push" => {
                    // tok_array_push(arr: PTR, tag: I64, data: I64) -> PTR
                    let arr = compile_expr(ctx, &args[0]).unwrap();
                    let val = compile_expr(ctx, &args[1]).unwrap();
                    let (tag, data) = to_tokvalue(ctx, val, &args[1].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_push");
                    let call = ctx.builder.ins().call(func_ref, &[arr, tag, data]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
                "tok_value_to_string" => {
                    // tok_value_to_string(tag: I64, data: I64) -> PTR
                    // HIR emits 1 arg but runtime expects (tag, data)
                    let val = compile_expr(ctx, &args[0]).unwrap();
                    let (tag, data) = to_tokvalue(ctx, val, &args[0].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_value_to_string");
                    let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
                "tok_array_concat" => {
                    // tok_array_concat(a: PTR, b: PTR) -> PTR
                    // Both args need to be unwrapped from Any if needed
                    let a_raw = compile_expr(ctx, &args[0]).unwrap();
                    let a = if matches!(&args[0].ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                        ctx.builder.ins().load(types::I64, MemFlags::trusted(), a_raw, 8)
                    } else {
                        a_raw
                    };
                    let b_raw = compile_expr(ctx, &args[1]).unwrap();
                    let b = if matches!(&args[1].ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                        ctx.builder.ins().load(types::I64, MemFlags::trusted(), b_raw, 8)
                    } else {
                        b_raw
                    };
                    let func_ref = ctx.get_runtime_func_ref("tok_array_concat");
                    let call = ctx.builder.ins().call(func_ref, &[a, b]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
                _ => {}
            }
            // Generic runtime call
            let mut arg_vals = Vec::new();
            for arg in args {
                if let Some(v) = compile_expr(ctx, arg) {
                    arg_vals.push(v);
                }
            }
            let func_ref = ctx.get_runtime_func_ref(name);
            let call = ctx.builder.ins().call(func_ref, &arg_vals);
            let results = ctx.builder.inst_results(call);
            if results.is_empty() {
                None
            } else {
                Some(results[0])
            }
        }

        HirExprKind::Lambda { params, ret_type, body } => {
            let lambda_name = format!("__tok_lambda_{}", ctx.compiler.lambda_counter);
            ctx.compiler.lambda_counter += 1;

            // ── Capture analysis ──────────────────────────────────
            // Collect free variables: names used in lambda body that aren't params
            let param_names: HashSet<String> = params.iter().map(|p| p.name.clone()).collect();
            let free_var_names = collect_free_vars(body, &param_names);

            // Filter to only variables that actually exist in the current scope,
            // and exclude user-defined functions (they're global, not captured)
            let mut captures: Vec<CapturedVar> = Vec::new();
            for name in &free_var_names {
                if let Some((_var, var_ty)) = ctx.vars.get(name) {
                    captures.push(CapturedVar {
                        name: name.clone(),
                        ty: var_ty.clone(),
                    });
                }
                // If not in vars, it's a global function or builtin — no capture needed
            }
            // Sort captures by name for deterministic ordering
            captures.sort_by(|a, b| a.name.cmp(&b.name));

            // Build uniform sig: (env: PTR, arg0_tag: I64, arg0_data: I64, ...) -> (I64, I64)
            let mut sig = ctx.compiler.module.make_signature();
            sig.params.push(AbiParam::new(PTR)); // env_ptr
            for _ in params {
                sig.params.push(AbiParam::new(types::I64)); // tag
                sig.params.push(AbiParam::new(types::I64)); // data
            }
            sig.returns.push(AbiParam::new(types::I64)); // result tag
            sig.returns.push(AbiParam::new(types::I64)); // result data

            let func_id = ctx.compiler.module
                .declare_function(&lambda_name, Linkage::Local, &sig)
                .unwrap();

            // Queue for later compilation (with captures info)
            let pending_idx = ctx.compiler.pending_lambdas.len();
            ctx.compiler.pending_lambdas.push(PendingLambda {
                name: lambda_name.clone(),
                func_id,
                params: params.clone(),
                ret_type: ret_type.clone(),
                body: body.clone(),
                captures: captures.clone(),
                specialized_param_types: None,
            });

            // Get fn pointer
            let func_ref = ctx.compiler.module
                .declare_func_in_func(func_id, ctx.builder.func);
            let fn_ptr = ctx.builder.ins().func_addr(PTR, func_ref);

            // ── Allocate environment and store captured values ────
            let env_ptr = if captures.is_empty() {
                ctx.builder.ins().iconst(PTR, 0) // null env
            } else {
                // Allocate env: count * 16 bytes (each capture is a TokValue)
                let count = ctx.builder.ins().iconst(types::I64, captures.len() as i64);
                let alloc_ref = ctx.get_runtime_func_ref("tok_env_alloc");
                let alloc_call = ctx.builder.ins().call(alloc_ref, &[count]);
                let env = ctx.builder.inst_results(alloc_call)[0];

                // Store each captured variable into the environment
                for (i, cap) in captures.iter().enumerate() {
                    let (var, var_ty) = ctx.vars.get(&cap.name).unwrap().clone();
                    let val = ctx.builder.use_var(var);
                    let (tag, data) = to_tokvalue(ctx, val, &var_ty);
                    let offset = (i * 16) as i32;
                    ctx.builder.ins().store(MemFlags::trusted(), tag, env, offset);
                    ctx.builder.ins().store(MemFlags::trusted(), data, env, offset + 8);
                }
                env
            };

            // Record lambda info for direct-call optimization
            ctx.last_lambda_info = Some((func_id, env_ptr, pending_idx));

            // Create closure: tok_closure_alloc(fn_ptr, env_ptr, arity)
            let arity = ctx.builder.ins().iconst(types::I32, params.len() as i64);
            let alloc_ref = ctx.get_runtime_func_ref("tok_closure_alloc");
            let call = ctx.builder.ins().call(alloc_ref, &[fn_ptr, env_ptr, arity]);
            Some(ctx.builder.inst_results(call)[0])
        }

        HirExprKind::If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            compile_if(ctx, cond, then_body, then_expr, else_body, else_expr, &expr.ty)
        }

        HirExprKind::Loop { kind, body } => {
            compile_loop(ctx, kind, body);
            None
        }

        HirExprKind::Block { stmts, expr: block_expr } => {
            for s in stmts {
                compile_stmt(ctx, s);
            }
            if let Some(e) = block_expr {
                compile_expr(ctx, e)
            } else {
                None
            }
        }

        HirExprKind::Length(target) => {
            let target_val = compile_expr(ctx, target).unwrap();
            match &target.ty {
                Type::Array(_) => {
                    let func_ref = ctx.get_runtime_func_ref("tok_array_len");
                    let call = ctx.builder.ins().call(func_ref, &[target_val]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                Type::Str => {
                    let func_ref = ctx.get_runtime_func_ref("tok_string_len");
                    let call = ctx.builder.ins().call(func_ref, &[target_val]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                Type::Map(_) => {
                    let func_ref = ctx.get_runtime_func_ref("tok_map_len");
                    let call = ctx.builder.ins().call(func_ref, &[target_val]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                Type::Tuple(_) => {
                    let func_ref = ctx.get_runtime_func_ref("tok_tuple_len");
                    let call = ctx.builder.ins().call(func_ref, &[target_val]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                Type::Any | Type::Optional(_) | Type::Result(_) => {
                    // Any: target_val is a pointer to a stack-allocated TokValue
                    // Extract (tag, data) and call tok_value_len
                    let (tag, data) = to_tokvalue(ctx, target_val, &target.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_value_len");
                    let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                _ => Some(ctx.builder.ins().iconst(types::I64, 0)),
            }
        }

        HirExprKind::Range { start, end: _, inclusive: _ } => {
            // Ranges are only used in for-loops, which handle them directly.
            // If a range appears as a standalone expression, just return start.
            compile_expr(ctx, start)
        }

        HirExprKind::Go(body_expr) => {
            // Goroutine — compile body as a thunk function and spawn via tok_go.
            // The thunk signature is: extern "C" fn(env: *mut u8) -> TokValue
            // which maps to Cranelift (PTR) -> (I64, I64)
            let thunk_name = format!("__tok_goroutine_{}", ctx.compiler.lambda_counter);
            ctx.compiler.lambda_counter += 1;

            // Capture analysis: find free variables in body that need to be passed to the thunk
            let empty_locals = HashSet::new();
            let mut free_set = HashSet::new();
            collect_free_vars_expr(body_expr, &empty_locals, &mut free_set);
            let free_var_names = free_set;
            let mut captures: Vec<CapturedVar> = Vec::new();
            for name in &free_var_names {
                if let Some((_var, var_ty)) = ctx.vars.get(name) {
                    captures.push(CapturedVar {
                        name: name.clone(),
                        ty: var_ty.clone(),
                    });
                }
            }
            captures.sort_by(|a, b| a.name.cmp(&b.name));

            // Declare thunk function: (env: PTR) -> (tag: I64, data: I64)
            let mut sig = ctx.compiler.module.make_signature();
            sig.params.push(AbiParam::new(PTR)); // env_ptr
            sig.returns.push(AbiParam::new(types::I64)); // result tag
            sig.returns.push(AbiParam::new(types::I64)); // result data

            let func_id = ctx.compiler.module
                .declare_function(&thunk_name, Linkage::Local, &sig)
                .unwrap();

            // Queue thunk for later compilation (reuse PendingLambda with 0 params)
            ctx.compiler.pending_lambdas.push(PendingLambda {
                name: thunk_name.clone(),
                func_id,
                params: vec![], // no params — it's a thunk
                ret_type: body_expr.ty.clone(),
                body: vec![HirStmt::Expr((**body_expr).clone())],
                captures: captures.clone(),
                specialized_param_types: None,
            });

            // Get thunk function pointer
            let func_ref = ctx.compiler.module
                .declare_func_in_func(func_id, ctx.builder.func);
            let fn_ptr = ctx.builder.ins().func_addr(PTR, func_ref);

            // Allocate environment for captures
            let env_ptr = if captures.is_empty() {
                ctx.builder.ins().iconst(PTR, 0)
            } else {
                let count = ctx.builder.ins().iconst(types::I64, captures.len() as i64);
                let alloc_ref = ctx.get_runtime_func_ref("tok_env_alloc");
                let alloc_call = ctx.builder.ins().call(alloc_ref, &[count]);
                let env = ctx.builder.inst_results(alloc_call)[0];

                for (i, cap) in captures.iter().enumerate() {
                    let (var, var_ty) = ctx.vars.get(&cap.name).unwrap().clone();
                    let val = ctx.builder.use_var(var);
                    let (tag, data) = to_tokvalue(ctx, val, &var_ty);
                    let offset = (i * 16) as i32;
                    ctx.builder.ins().store(MemFlags::trusted(), tag, env, offset);
                    ctx.builder.ins().store(MemFlags::trusted(), data, env, offset + 8);
                }
                env
            };

            // Call tok_go(fn_ptr, env_ptr) -> *mut TokHandle
            let go_ref = ctx.get_runtime_func_ref("tok_go");
            let call = ctx.builder.ins().call(go_ref, &[fn_ptr, env_ptr]);
            Some(ctx.builder.inst_results(call)[0])
        }

        HirExprKind::Receive(chan_expr) => {
            let chan = compile_expr(ctx, chan_expr).unwrap();
            // Distinguish channel recv from handle join based on expression type
            match &chan_expr.ty {
                Type::Handle(_) => {
                    // Join the goroutine handle
                    let func_ref = ctx.get_runtime_func_ref("tok_handle_join");
                    let call = ctx.builder.ins().call(func_ref, &[chan]);
                    let results = ctx.builder.inst_results(call);
                    Some(from_tokvalue(ctx, results[0], results[1], &expr.ty))
                }
                _ => {
                    // Channel receive
                    let func_ref = ctx.get_runtime_func_ref("tok_channel_recv");
                    let call = ctx.builder.ins().call(func_ref, &[chan]);
                    let results = ctx.builder.inst_results(call);
                    Some(from_tokvalue(ctx, results[0], results[1], &expr.ty))
                }
            }
        }

        HirExprKind::Send { chan, value } => {
            let chan_val = compile_expr(ctx, chan).unwrap();
            let val = compile_expr(ctx, value).unwrap();
            let (tag, data) = to_tokvalue(ctx, val, &value.ty);
            let func_ref = ctx.get_runtime_func_ref("tok_channel_send");
            ctx.builder.ins().call(func_ref, &[chan_val, tag, data]);
            None
        }

        HirExprKind::Select(arms) => {
            // Simplified select: try each arm in order.
            // Full implementation would use runtime select.
            // For now, just compile the first arm's body.
            if let Some(arm) = arms.first() {
                match arm {
                    HirSelectArm::Recv { var, chan, body } => {
                        let chan_val = compile_expr(ctx, chan).unwrap();
                        let func_ref = ctx.get_runtime_func_ref("tok_channel_recv");
                        let call = ctx.builder.ins().call(func_ref, &[chan_val]);
                        let results = ctx.builder.inst_results(call);
                        let val = results[1]; // data word
                        let ct = cl_type_or_i64(&Type::Any);
                        let v = ctx.new_var(ct);
                        ctx.builder.def_var(v, val);
                        ctx.vars.insert(var.clone(), (v, Type::Any));
                        compile_body(ctx, body, &Type::Nil);
                    }
                    HirSelectArm::Default(body) => {
                        compile_body(ctx, body, &Type::Nil);
                    }
                    HirSelectArm::Send { chan, value, body } => {
                        let chan_val = compile_expr(ctx, chan).unwrap();
                        let val = compile_expr(ctx, value).unwrap();
                        let (tag, data) = to_tokvalue(ctx, val, &Type::Any);
                        let func_ref = ctx.get_runtime_func_ref("tok_channel_send");
                        ctx.builder.ins().call(func_ref, &[chan_val, tag, data]);
                        compile_body(ctx, body, &Type::Nil);
                    }
                }
            }
            None
        }
    }
}

// ─── Binary operators ─────────────────────────────────────────────────

fn compile_binop(
    ctx: &mut FuncCtx,
    op: HirBinOp,
    left: &HirExpr,
    right: &HirExpr,
    result_ty: &Type,
) -> Option<Value> {
    // Short-circuit for And/Or
    match op {
        HirBinOp::And => return compile_short_circuit_and(ctx, left, right, result_ty),
        HirBinOp::Or => return compile_short_circuit_or(ctx, left, right, result_ty),
        _ => {}
    }

    let lv = compile_expr(ctx, left).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
    let rv = compile_expr(ctx, right).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));

    // If both sides are Int
    if matches!(left.ty, Type::Int) && matches!(right.ty, Type::Int) {
        return compile_int_binop(ctx, op, lv, rv);
    }

    // If both sides are Float (or one is Float and one is Int)
    if matches!(left.ty, Type::Float) || matches!(right.ty, Type::Float) {
        let lf = if matches!(left.ty, Type::Int) {
            ctx.builder.ins().fcvt_from_sint(types::F64, lv)
        } else {
            lv
        };
        let rf = if matches!(right.ty, Type::Int) {
            ctx.builder.ins().fcvt_from_sint(types::F64, rv)
        } else {
            rv
        };
        return compile_float_binop(ctx, op, lf, rf);
    }

    // String concatenation
    if matches!(left.ty, Type::Str) && matches!(right.ty, Type::Str) && matches!(op, HirBinOp::Add) {
        let func_ref = ctx.get_runtime_func_ref("tok_string_concat");
        let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
        return Some(ctx.builder.inst_results(call)[0]);
    }

    // String comparison
    if matches!(left.ty, Type::Str) && matches!(right.ty, Type::Str) {
        match op {
            HirBinOp::Eq => {
                let func_ref = ctx.get_runtime_func_ref("tok_string_eq");
                let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
                return Some(ctx.builder.inst_results(call)[0]);
            }
            HirBinOp::Neq => {
                let func_ref = ctx.get_runtime_func_ref("tok_string_eq");
                let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
                let eq = ctx.builder.inst_results(call)[0];
                let one = ctx.builder.ins().iconst(types::I8, 1);
                return Some(ctx.builder.ins().bxor(eq, one));
            }
            HirBinOp::Lt | HirBinOp::Gt | HirBinOp::LtEq | HirBinOp::GtEq => {
                let func_ref = ctx.get_runtime_func_ref("tok_string_cmp");
                let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
                let cmp = ctx.builder.inst_results(call)[0];
                let zero = ctx.builder.ins().iconst(types::I64, 0);
                let cc = match op {
                    HirBinOp::Lt => cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                    HirBinOp::Gt => cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThan,
                    HirBinOp::LtEq => cranelift_codegen::ir::condcodes::IntCC::SignedLessThanOrEqual,
                    HirBinOp::GtEq => cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThanOrEqual,
                    _ => unreachable!(),
                };
                let result = ctx.builder.ins().icmp(cc, cmp, zero);
                return Some(result);
            }
            _ => {}
        }
    }

    // Bool comparisons
    if matches!(left.ty, Type::Bool) && matches!(right.ty, Type::Bool) {
        match op {
            HirBinOp::Eq => {
                let result = ctx.builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::Equal,
                    lv, rv,
                );
                return Some(result);
            }
            HirBinOp::Neq => {
                let result = ctx.builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                    lv, rv,
                );
                return Some(result);
            }
            _ => {}
        }
    }

    // Fallback: use runtime value ops for Any types
    let (lt, ld) = to_tokvalue(ctx, lv, &left.ty);
    let (rt, rd) = to_tokvalue(ctx, rv, &right.ty);
    let rt_name = match op {
        HirBinOp::Add => "tok_value_add",
        HirBinOp::Sub => "tok_value_sub",
        HirBinOp::Mul => "tok_value_mul",
        HirBinOp::Div => "tok_value_div",
        HirBinOp::Mod => "tok_value_mod",
        HirBinOp::Eq => {
            let func_ref = ctx.get_runtime_func_ref("tok_value_eq");
            let call = ctx.builder.ins().call(func_ref, &[lt, ld, rt, rd]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
        HirBinOp::Neq => {
            let func_ref = ctx.get_runtime_func_ref("tok_value_eq");
            let call = ctx.builder.ins().call(func_ref, &[lt, ld, rt, rd]);
            let eq = ctx.builder.inst_results(call)[0];
            let one = ctx.builder.ins().iconst(types::I8, 1);
            return Some(ctx.builder.ins().bxor(eq, one));
        }
        HirBinOp::Lt => {
            let func_ref = ctx.get_runtime_func_ref("tok_value_lt");
            let call = ctx.builder.ins().call(func_ref, &[lt, ld, rt, rd]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
        HirBinOp::Gt => {
            let func_ref = ctx.get_runtime_func_ref("tok_value_lt");
            let call = ctx.builder.ins().call(func_ref, &[rt, rd, lt, ld]); // swap
            return Some(ctx.builder.inst_results(call)[0]);
        }
        HirBinOp::LtEq => {
            // a <= b = !(b < a)
            let func_ref = ctx.get_runtime_func_ref("tok_value_lt");
            let call = ctx.builder.ins().call(func_ref, &[rt, rd, lt, ld]);
            let lt_result = ctx.builder.inst_results(call)[0];
            let one = ctx.builder.ins().iconst(types::I8, 1);
            return Some(ctx.builder.ins().bxor(lt_result, one));
        }
        HirBinOp::GtEq => {
            // a >= b = !(a < b)
            let func_ref = ctx.get_runtime_func_ref("tok_value_lt");
            let call = ctx.builder.ins().call(func_ref, &[lt, ld, rt, rd]);
            let lt_result = ctx.builder.inst_results(call)[0];
            let one = ctx.builder.ins().iconst(types::I8, 1);
            return Some(ctx.builder.ins().bxor(lt_result, one));
        }
        _ => {
            // Bitwise ops, Pow — fallback to 0
            return Some(ctx.builder.ins().iconst(types::I64, 0));
        }
    };
    let func_ref = ctx.get_runtime_func_ref(rt_name);
    let call = ctx.builder.ins().call(func_ref, &[lt, ld, rt, rd]);
    let results = ctx.builder.inst_results(call);
    Some(from_tokvalue(ctx, results[0], results[1], result_ty))
}

fn compile_int_binop(
    ctx: &mut FuncCtx,
    op: HirBinOp,
    lv: Value,
    rv: Value,
) -> Option<Value> {
    use cranelift_codegen::ir::condcodes::IntCC;
    Some(match op {
        HirBinOp::Add => ctx.builder.ins().iadd(lv, rv),
        HirBinOp::Sub => ctx.builder.ins().isub(lv, rv),
        HirBinOp::Mul => ctx.builder.ins().imul(lv, rv),
        HirBinOp::Div => ctx.builder.ins().sdiv(lv, rv),
        HirBinOp::Mod => ctx.builder.ins().srem(lv, rv),
        HirBinOp::Pow => {
            // Integer power — use a loop or runtime call.
            // Simple approach: convert to float, pow, convert back.
            let _lf = ctx.builder.ins().fcvt_from_sint(types::F64, lv);
            let _rf = ctx.builder.ins().fcvt_from_sint(types::F64, rv);
            // No fpow in Cranelift — use a simple loop for integer exponentiation.
            // For now, just multiply (only handles small cases).
            // TODO: runtime call for general pow
            ctx.builder.ins().imul(lv, rv) // placeholder
        }
        HirBinOp::Eq => ctx.builder.ins().icmp(IntCC::Equal, lv, rv),
        HirBinOp::Neq => ctx.builder.ins().icmp(IntCC::NotEqual, lv, rv),
        HirBinOp::Lt => ctx.builder.ins().icmp(IntCC::SignedLessThan, lv, rv),
        HirBinOp::Gt => ctx.builder.ins().icmp(IntCC::SignedGreaterThan, lv, rv),
        HirBinOp::LtEq => ctx.builder.ins().icmp(IntCC::SignedLessThanOrEqual, lv, rv),
        HirBinOp::GtEq => ctx.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, lv, rv),
        HirBinOp::BitAnd => ctx.builder.ins().band(lv, rv),
        HirBinOp::BitOr => ctx.builder.ins().bor(lv, rv),
        HirBinOp::BitXor => ctx.builder.ins().bxor(lv, rv),
        HirBinOp::Shl => ctx.builder.ins().ishl(lv, rv),
        HirBinOp::Shr => ctx.builder.ins().sshr(lv, rv),
        HirBinOp::And | HirBinOp::Or => unreachable!("handled by short-circuit"),
    })
}

fn compile_float_binop(
    ctx: &mut FuncCtx,
    op: HirBinOp,
    lv: Value,
    rv: Value,
) -> Option<Value> {
    use cranelift_codegen::ir::condcodes::FloatCC;
    Some(match op {
        HirBinOp::Add => ctx.builder.ins().fadd(lv, rv),
        HirBinOp::Sub => ctx.builder.ins().fsub(lv, rv),
        HirBinOp::Mul => ctx.builder.ins().fmul(lv, rv),
        HirBinOp::Div => ctx.builder.ins().fdiv(lv, rv),
        HirBinOp::Mod => {
            // fmod: a - floor(a/b) * b
            let div = ctx.builder.ins().fdiv(lv, rv);
            let floored = ctx.builder.ins().floor(div);
            let prod = ctx.builder.ins().fmul(floored, rv);
            ctx.builder.ins().fsub(lv, prod)
        }
        HirBinOp::Pow => {
            // No fpow in Cranelift. Use runtime.
            // For now just return lv as placeholder.
            // TODO: link libm or implement pow
            lv
        }
        HirBinOp::Eq => ctx.builder.ins().fcmp(FloatCC::Equal, lv, rv),
        HirBinOp::Neq => ctx.builder.ins().fcmp(FloatCC::NotEqual, lv, rv),
        HirBinOp::Lt => ctx.builder.ins().fcmp(FloatCC::LessThan, lv, rv),
        HirBinOp::Gt => ctx.builder.ins().fcmp(FloatCC::GreaterThan, lv, rv),
        HirBinOp::LtEq => ctx.builder.ins().fcmp(FloatCC::LessThanOrEqual, lv, rv),
        HirBinOp::GtEq => ctx.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lv, rv),
        _ => lv, // bitwise ops on float don't make sense
    })
}

// ─── Short-circuit logic ──────────────────────────────────────────────

fn compile_short_circuit_and(
    ctx: &mut FuncCtx,
    left: &HirExpr,
    right: &HirExpr,
    _result_ty: &Type,
) -> Option<Value> {
    let lv = compile_expr(ctx, left).unwrap();
    let then_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();
    ctx.builder.append_block_param(merge_block, types::I8);

    // If left is falsy, short-circuit to false
    let cond = to_bool(ctx, lv, &left.ty);
    let false_val = ctx.builder.ins().iconst(types::I8, 0);
    ctx.builder.ins().brif(cond, then_block, &[], merge_block, &[false_val]);

    ctx.builder.switch_to_block(then_block);
    ctx.builder.seal_block(then_block);
    let rv = compile_expr(ctx, right).unwrap();
    let right_bool = to_bool(ctx, rv, &right.ty);
    ctx.builder.ins().jump(merge_block, &[right_bool]);

    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
    Some(ctx.builder.block_params(merge_block)[0])
}

fn compile_short_circuit_or(
    ctx: &mut FuncCtx,
    left: &HirExpr,
    right: &HirExpr,
    _result_ty: &Type,
) -> Option<Value> {
    let lv = compile_expr(ctx, left).unwrap();
    let else_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();
    ctx.builder.append_block_param(merge_block, types::I8);

    let cond = to_bool(ctx, lv, &left.ty);
    let true_val = ctx.builder.ins().iconst(types::I8, 1);
    ctx.builder.ins().brif(cond, merge_block, &[true_val], else_block, &[]);

    ctx.builder.switch_to_block(else_block);
    ctx.builder.seal_block(else_block);
    let rv = compile_expr(ctx, right).unwrap();
    let right_bool = to_bool(ctx, rv, &right.ty);
    ctx.builder.ins().jump(merge_block, &[right_bool]);

    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
    Some(ctx.builder.block_params(merge_block)[0])
}

// ─── Unary operators ──────────────────────────────────────────────────

fn compile_unaryop(
    ctx: &mut FuncCtx,
    op: HirUnaryOp,
    operand: &HirExpr,
    _result_ty: &Type,
) -> Option<Value> {
    let val = compile_expr(ctx, operand).unwrap();
    match op {
        HirUnaryOp::Neg => {
            if matches!(operand.ty, Type::Int) {
                Some(ctx.builder.ins().ineg(val))
            } else if matches!(operand.ty, Type::Float) {
                Some(ctx.builder.ins().fneg(val))
            } else {
                let (tag, data) = to_tokvalue(ctx, val, &operand.ty);
                let func_ref = ctx.get_runtime_func_ref("tok_value_negate");
                let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                let results = ctx.builder.inst_results(call);
                Some(from_tokvalue(ctx, results[0], results[1], &operand.ty))
            }
        }
        HirUnaryOp::Not => {
            if matches!(operand.ty, Type::Bool) {
                let one = ctx.builder.ins().iconst(types::I8, 1);
                Some(ctx.builder.ins().bxor(val, one))
            } else {
                let (tag, data) = to_tokvalue(ctx, val, &operand.ty);
                let func_ref = ctx.get_runtime_func_ref("tok_value_not");
                let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                Some(ctx.builder.inst_results(call)[0])
            }
        }
    }
}

// ─── Function calls ───────────────────────────────────────────────────

fn compile_call(
    ctx: &mut FuncCtx,
    func_expr: &HirExpr,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    // Check if this is a call to a known function name
    if let HirExprKind::Ident(name) = &func_expr.kind {
        // User-defined functions take priority over builtins
        if ctx.compiler.declared_funcs.contains_key(name.as_str()) {
            return compile_user_func_call(ctx, name, args, result_ty);
        }
        // Built-in function calls
        match name.as_str() {
            "p" | "print" => {
                return compile_print_call(ctx, args, false);
            }
            "pl" | "println" => {
                return compile_print_call(ctx, args, true);
            }
            "len" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    match &arg.ty {
                        Type::Any | Type::Optional(_) | Type::Result(_) => {
                            let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                            let func_ref = ctx.get_runtime_func_ref("tok_value_len");
                            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                            return Some(ctx.builder.inst_results(call)[0]);
                        }
                        _ => {
                            let func_name = match &arg.ty {
                                Type::Array(_) => "tok_array_len",
                                Type::Str => "tok_string_len",
                                Type::Map(_) => "tok_map_len",
                                Type::Tuple(_) => "tok_tuple_len",
                                _ => return Some(ctx.builder.ins().iconst(types::I64, 0)),
                            };
                            let func_ref = ctx.get_runtime_func_ref(func_name);
                            let call = ctx.builder.ins().call(func_ref, &[val]);
                            return Some(ctx.builder.inst_results(call)[0]);
                        }
                    }
                }
            }
            "push" => {
                if args.len() >= 2 {
                    let arr_raw = compile_expr(ctx, &args[0]).unwrap();
                    let arr = unwrap_any_ptr(ctx, arr_raw, &args[0].ty);
                    let val = compile_expr(ctx, &args[1]).unwrap();
                    let (tag, data) = to_tokvalue(ctx, val, &args[1].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_push");
                    let call = ctx.builder.ins().call(func_ref, &[arr, tag, data]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "sort" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    let ptr = unwrap_any_ptr(ctx, val, &arg.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_sort");
                    let call = ctx.builder.ins().call(func_ref, &[ptr]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "rev" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    let ptr = unwrap_any_ptr(ctx, val, &arg.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_rev");
                    let call = ctx.builder.ins().call(func_ref, &[ptr]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "flat" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    let ptr = unwrap_any_ptr(ctx, val, &arg.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_flat");
                    let call = ctx.builder.ins().call(func_ref, &[ptr]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "uniq" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    let ptr = unwrap_any_ptr(ctx, val, &arg.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_uniq");
                    let call = ctx.builder.ins().call(func_ref, &[ptr]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "join" => {
                if args.len() >= 2 {
                    let arr_raw = compile_expr(ctx, &args[0]).unwrap();
                    let arr = unwrap_any_ptr(ctx, arr_raw, &args[0].ty);
                    let sep_raw = compile_expr(ctx, &args[1]).unwrap();
                    let sep = unwrap_any_ptr(ctx, sep_raw, &args[1].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_join");
                    let call = ctx.builder.ins().call(func_ref, &[arr, sep]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "split" => {
                if args.len() >= 2 {
                    let s_raw = compile_expr(ctx, &args[0]).unwrap();
                    let s = unwrap_any_ptr(ctx, s_raw, &args[0].ty);
                    let delim_raw = compile_expr(ctx, &args[1]).unwrap();
                    let delim = unwrap_any_ptr(ctx, delim_raw, &args[1].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_string_split");
                    let call = ctx.builder.ins().call(func_ref, &[s, delim]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "trim" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    let ptr = unwrap_any_ptr(ctx, val, &arg.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_string_trim");
                    let call = ctx.builder.ins().call(func_ref, &[ptr]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "keys" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    let ptr = unwrap_any_ptr(ctx, val, &arg.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_map_keys");
                    let call = ctx.builder.ins().call(func_ref, &[ptr]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "vals" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    let ptr = unwrap_any_ptr(ctx, val, &arg.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_map_vals");
                    let call = ctx.builder.ins().call(func_ref, &[ptr]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "has" => {
                if args.len() >= 2 {
                    let map_raw = compile_expr(ctx, &args[0]).unwrap();
                    let map = unwrap_any_ptr(ctx, map_raw, &args[0].ty);
                    let key_raw = compile_expr(ctx, &args[1]).unwrap();
                    let key = unwrap_any_ptr(ctx, key_raw, &args[1].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_map_has");
                    let call = ctx.builder.ins().call(func_ref, &[map, key]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "del" => {
                if args.len() >= 2 {
                    let map_raw = compile_expr(ctx, &args[0]).unwrap();
                    let map = unwrap_any_ptr(ctx, map_raw, &args[0].ty);
                    let key_raw = compile_expr(ctx, &args[1]).unwrap();
                    let key = unwrap_any_ptr(ctx, key_raw, &args[1].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_map_del");
                    let call = ctx.builder.ins().call(func_ref, &[map, key]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "int" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    if matches!(arg.ty, Type::Int) {
                        return Some(val);
                    }
                    if matches!(arg.ty, Type::Float) {
                        return Some(ctx.builder.ins().fcvt_to_sint(types::I64, val));
                    }
                    let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_to_int");
                    let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "float" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    if matches!(arg.ty, Type::Float) {
                        return Some(val);
                    }
                    if matches!(arg.ty, Type::Int) {
                        return Some(ctx.builder.ins().fcvt_from_sint(types::F64, val));
                    }
                    let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_to_float");
                    let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "str" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    let func_name = match &arg.ty {
                        Type::Int => "tok_int_to_string",
                        Type::Float => "tok_float_to_string",
                        Type::Bool => "tok_bool_to_string",
                        Type::Str => return Some(val),
                        _ => {
                            let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                            let func_ref = ctx.get_runtime_func_ref("tok_value_to_string");
                            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                            return Some(ctx.builder.inst_results(call)[0]);
                        }
                    };
                    let func_ref = ctx.get_runtime_func_ref(func_name);
                    let call = ctx.builder.ins().call(func_ref, &[val]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "abs" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    if matches!(arg.ty, Type::Any) {
                        // Fully dynamic: dispatch at runtime
                        let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                        let func_ref = ctx.get_runtime_func_ref("tok_value_abs");
                        let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                        let results = ctx.builder.inst_results(call);
                        return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
                    }
                    let is_float = matches!(arg.ty, Type::Float);
                    let func_name = if is_float { "tok_abs_float" } else { "tok_abs_int" };
                    let func_ref = ctx.get_runtime_func_ref(func_name);
                    let call = ctx.builder.ins().call(func_ref, &[val]);
                    let raw = ctx.builder.inst_results(call)[0];
                    // Wrap if caller expects Any
                    if matches!(result_ty, Type::Any) {
                        let tag_val = ctx.builder.ins().iconst(types::I64,
                            if is_float { TAG_FLOAT } else { TAG_INT } as i64);
                        let data_val = if is_float {
                            ctx.builder.ins().bitcast(types::I64, MemFlags::new(), raw)
                        } else { raw };
                        return Some(alloc_tokvalue_on_stack(ctx, tag_val, data_val));
                    }
                    return Some(raw);
                }
            }
            "floor" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    if matches!(arg.ty, Type::Any) {
                        let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                        let func_ref = ctx.get_runtime_func_ref("tok_value_floor");
                        let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                        let results = ctx.builder.inst_results(call);
                        return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
                    }
                    let func_ref = ctx.get_runtime_func_ref("tok_floor");
                    let call = ctx.builder.ins().call(func_ref, &[val]);
                    let raw = ctx.builder.inst_results(call)[0];
                    if matches!(result_ty, Type::Any) {
                        let tag_val = ctx.builder.ins().iconst(types::I64, TAG_INT as i64);
                        return Some(alloc_tokvalue_on_stack(ctx, tag_val, raw));
                    }
                    return Some(raw);
                }
            }
            "ceil" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    if matches!(arg.ty, Type::Any) {
                        let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                        let func_ref = ctx.get_runtime_func_ref("tok_value_ceil");
                        let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                        let results = ctx.builder.inst_results(call);
                        return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
                    }
                    let func_ref = ctx.get_runtime_func_ref("tok_ceil");
                    let call = ctx.builder.ins().call(func_ref, &[val]);
                    let raw = ctx.builder.inst_results(call)[0];
                    if matches!(result_ty, Type::Any) {
                        let tag_val = ctx.builder.ins().iconst(types::I64, TAG_INT as i64);
                        return Some(alloc_tokvalue_on_stack(ctx, tag_val, raw));
                    }
                    return Some(raw);
                }
            }
            "rand" => {
                let func_ref = ctx.get_runtime_func_ref("tok_rand");
                let call = ctx.builder.ins().call(func_ref, &[]);
                return Some(ctx.builder.inst_results(call)[0]);
            }
            "clock" => {
                let func_ref = ctx.get_runtime_func_ref("tok_clock");
                let call = ctx.builder.ins().call(func_ref, &[]);
                return Some(ctx.builder.inst_results(call)[0]);
            }
            "exit" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    let func_ref = ctx.get_runtime_func_ref("tok_exit");
                    ctx.builder.ins().call(func_ref, &[val]);
                }
                return None;
            }
            "chan" => {
                let cap = if let Some(arg) = args.first() {
                    compile_expr(ctx, arg).unwrap()
                } else {
                    ctx.builder.ins().iconst(types::I64, 0)
                };
                let func_ref = ctx.get_runtime_func_ref("tok_channel_alloc");
                let call = ctx.builder.ins().call(func_ref, &[cap]);
                return Some(ctx.builder.inst_results(call)[0]);
            }
            "type" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).unwrap();
                    let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_type_of");
                    let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "min" => {
                if args.len() == 1 {
                    let val = compile_expr(ctx, &args[0]).unwrap();
                    let func_ref = ctx.get_runtime_func_ref("tok_array_min");
                    let call = ctx.builder.ins().call(func_ref, &[val]);
                    let results = ctx.builder.inst_results(call);
                    return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
                }
            }
            "max" => {
                if args.len() == 1 {
                    let val = compile_expr(ctx, &args[0]).unwrap();
                    let ptr = unwrap_any_ptr(ctx, val, &args[0].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_max");
                    let call = ctx.builder.ins().call(func_ref, &[ptr]);
                    let results = ctx.builder.inst_results(call);
                    return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
                }
            }
            "sum" => {
                if args.len() == 1 {
                    let val = compile_expr(ctx, &args[0]).unwrap();
                    let ptr = unwrap_any_ptr(ctx, val, &args[0].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_sum");
                    let call = ctx.builder.ins().call(func_ref, &[ptr]);
                    let results = ctx.builder.inst_results(call);
                    return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
                }
            }
            "slice" => {
                if args.len() >= 3 {
                    let target = compile_expr(ctx, &args[0]).unwrap();
                    let start = compile_expr(ctx, &args[1]).unwrap();
                    let end = compile_expr(ctx, &args[2]).unwrap();
                    let func_name = match &args[0].ty {
                        Type::Array(_) => "tok_array_slice",
                        Type::Str => "tok_string_slice",
                        _ => return None,
                    };
                    let func_ref = ctx.get_runtime_func_ref(func_name);
                    let call = ctx.builder.ins().call(func_ref, &[target, start, end]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "pmap" => {
                if args.len() >= 2 {
                    let arr_raw = compile_expr(ctx, &args[0]).unwrap();
                    let arr = unwrap_any_ptr(ctx, arr_raw, &args[0].ty);
                    let closure = compile_expr(ctx, &args[1]).unwrap();
                    let closure_ptr = unwrap_any_ptr(ctx, closure, &args[1].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_pmap");
                    let call = ctx.builder.ins().call(func_ref, &[arr, closure_ptr]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            _ => {
                // Direct call optimization: if we know the FuncId, skip indirect dispatch
                if let Some(kc) = ctx.known_closures.get(name).cloned() {
                    // Try specialized call if all arg types are concrete
                    let arg_types: Vec<Type> = args.iter().map(|a| a.ty.clone()).collect();
                    let all_concrete = arg_types.iter().all(|t| matches!(t, Type::Int | Type::Float | Type::Bool));
                    if all_concrete {
                        return compile_specialized_closure_call(ctx, name, &kc, args, &arg_types, result_ty);
                    }
                    return compile_direct_closure_call(ctx, kc.func_id, kc.env_ptr, args, result_ty);
                }
                // Check if it's a variable holding a closure
                if let Some((var, var_ty)) = ctx.vars.get(name).cloned() {
                    if matches!(var_ty, Type::Func(_)) {
                        let closure_ptr = ctx.builder.use_var(var);
                        return compile_closure_call(ctx, closure_ptr, args, result_ty);
                    }
                    if matches!(var_ty, Type::Any) {
                        // Any-typed variable might hold a closure — extract ptr from TokValue data field
                        let tokval_ptr = ctx.builder.use_var(var);
                        let closure_ptr = ctx.builder.ins().load(types::I64, MemFlags::trusted(), tokval_ptr, 8);
                        return compile_closure_call(ctx, closure_ptr, args, result_ty);
                    }
                }
                // Unknown — fall through to generic call below
            }
        }
    }

    // Generic function call (through closure expression)
    let func_val = compile_expr(ctx, func_expr);
    if let Some(closure_ptr) = func_val {
        return compile_closure_call(ctx, closure_ptr, args, result_ty);
    }

    Some(ctx.builder.ins().iconst(types::I64, 0))
}

fn compile_user_func_call(
    ctx: &mut FuncCtx,
    name: &str,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    let func_sig = ctx.compiler.func_sigs.get(name).cloned();
    let mut arg_vals = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        if let Some(v) = compile_expr(ctx, arg) {
            let param_ty = func_sig.as_ref()
                .and_then(|s| s.0.get(i))
                .cloned()
                .unwrap_or(arg.ty.clone());
            if matches!(param_ty, Type::Any) {
                // Any param: always pass as (tag, data) pair
                let (tag, data) = to_tokvalue(ctx, v, &arg.ty);
                arg_vals.push(tag);
                arg_vals.push(data);
            } else if matches!(arg.ty, Type::Any) {
                // Any → Concrete param: extract from TokValue ptr
                let coerced = coerce_value(ctx, v, &arg.ty, &param_ty);
                arg_vals.push(coerced);
            } else {
                arg_vals.push(v);
            }
        }
    }
    let func_ref = ctx.get_tok_func_ref(name);
    let call = ctx.builder.ins().call(func_ref, &arg_vals);
    let results = ctx.builder.inst_results(call);
    if results.is_empty() {
        return None;
    }
    let ret_ty = func_sig.as_ref()
        .map(|s| s.1.clone())
        .unwrap_or(result_ty.clone());
    if matches!(ret_ty, Type::Any) {
        // Any return: 2 results (tag, data) — pack into stack TokValue
        let tag = results[0];
        let data = results[1];
        return Some(alloc_tokvalue_on_stack(ctx, tag, data));
    }
    Some(coerce_value(ctx, results[0], &ret_ty, result_ty))
}

/// Call a closure indirectly: extract fn_ptr and env_ptr, build signature, call_indirect.
fn compile_closure_call(
    ctx: &mut FuncCtx,
    closure_ptr: Value,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    // Extract fn_ptr and env_ptr directly from TokClosure struct (repr(C)):
    //   +0: rc (AtomicU32, 4B) + padding (4B)
    //   +8: fn_ptr (*const u8, 8B)
    //   +16: env_ptr (*mut u8, 8B)
    let fn_ptr = ctx.builder.ins().load(PTR, MemFlags::trusted(), closure_ptr, 8);
    let env_ptr = ctx.builder.ins().load(PTR, MemFlags::trusted(), closure_ptr, 16);

    // Build signature for indirect call: (env: PTR, tag0: I64, data0: I64, ...) -> (I64, I64)
    let mut sig = ctx.compiler.module.make_signature();
    sig.params.push(AbiParam::new(PTR)); // env
    for _ in args {
        sig.params.push(AbiParam::new(types::I64)); // tag
        sig.params.push(AbiParam::new(types::I64)); // data
    }
    sig.returns.push(AbiParam::new(types::I64)); // ret tag
    sig.returns.push(AbiParam::new(types::I64)); // ret data
    let sig_ref = ctx.builder.import_signature(sig);

    // Build args: env, then (tag, data) pairs for each arg
    let mut call_args = vec![env_ptr];
    for arg in args {
        let v = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        let (tag, data) = to_tokvalue(ctx, v, &arg.ty);
        call_args.push(tag);
        call_args.push(data);
    }

    let call = ctx.builder.ins().call_indirect(sig_ref, fn_ptr, &call_args);
    let results = ctx.builder.inst_results(call);
    Some(from_tokvalue(ctx, results[0], results[1], result_ty))
}

/// Call a closure directly when we know the FuncId at compile time.
/// Still uses the uniform (tag, data) calling convention, but avoids call_indirect.
fn compile_direct_closure_call(
    ctx: &mut FuncCtx,
    func_id: FuncId,
    env_ptr: Value,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    let func_ref = ctx.compiler.module
        .declare_func_in_func(func_id, ctx.builder.func);

    // Build args: env, then (tag, data) pairs for each arg
    let mut call_args = vec![env_ptr];
    for arg in args {
        let v = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        let (tag, data) = to_tokvalue(ctx, v, &arg.ty);
        call_args.push(tag);
        call_args.push(data);
    }

    let call = ctx.builder.ins().call(func_ref, &call_args);
    let results = ctx.builder.inst_results(call);
    Some(from_tokvalue(ctx, results[0], results[1], result_ty))
}

/// Call a closure with a type-specialized calling convention (native types, no boxing).
/// Lazily creates the specialized function on first call.
fn compile_specialized_closure_call(
    ctx: &mut FuncCtx,
    name: &str,
    kc: &KnownClosure,
    args: &[HirExpr],
    arg_types: &[Type],
    result_ty: &Type,
) -> Option<Value> {
    // Check if we already have a specialized version for these arg types
    let existing = if let Some((sid, ref stypes, ref sret)) = kc.specialized {
        if stypes == arg_types { Some((sid, sret.clone())) } else { None }
    } else {
        None
    };

    let (spec_func_id, spec_ret_type) = if let Some(pair) = existing {
        pair
    } else {
        // Create specialized function
        let orig = &ctx.compiler.pending_lambdas[kc.pending_idx];
        let spec_name = format!("{}_spec", orig.name);

        // Build type map and retype body FIRST so we can get the accurate return type
        let mut type_map = HashMap::new();
        for (param, at) in orig.params.iter().zip(arg_types.iter()) {
            type_map.insert(param.name.clone(), at.clone());
        }
        for cap in &orig.captures {
            if matches!(cap.ty, Type::Int | Type::Float | Type::Bool) {
                type_map.insert(cap.name.clone(), cap.ty.clone());
            }
        }
        let retyped_body = retype_body(&orig.body, &type_map);

        // Derive return type from the retyped body's last expression
        let ret_type = retyped_body.last()
            .and_then(|s| match s {
                HirStmt::Expr(e) => if matches!(e.ty, Type::Any) { None } else { Some(e.ty.clone()) },
                _ => None,
            })
            .unwrap_or(Type::Int); // fallback for simple arithmetic lambdas

        // Build signature: (env: PTR, arg0: T0, ...) -> RetT
        let mut sig = ctx.compiler.module.make_signature();
        sig.params.push(AbiParam::new(PTR)); // env_ptr
        for at in arg_types {
            sig.params.push(AbiParam::new(cl_type_or_i64(at)));
        }
        sig.returns.push(AbiParam::new(cl_type_or_i64(&ret_type)));

        let func_id = ctx.compiler.module
            .declare_function(&spec_name, Linkage::Local, &sig)
            .unwrap();

        // Create specialized PendingLambda with retyped body
        ctx.compiler.pending_lambdas.push(PendingLambda {
            name: spec_name,
            func_id,
            params: orig.params.clone(),
            ret_type: ret_type.clone(),
            body: retyped_body,
            captures: orig.captures.clone(),
            specialized_param_types: Some(arg_types.to_vec()),
        });

        // Update known_closures with specialized info
        if let Some(kc_mut) = ctx.known_closures.get_mut(name) {
            kc_mut.specialized = Some((func_id, arg_types.to_vec(), ret_type.clone()));
        }

        (func_id, ret_type)
    };

    let env_ptr = kc.env_ptr;
    let func_ref = ctx.compiler.module
        .declare_func_in_func(spec_func_id, ctx.builder.func);

    // Build args with native types (no boxing)
    let mut call_args = vec![env_ptr];
    for arg in args {
        let v = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        call_args.push(v);
    }

    let call = ctx.builder.ins().call(func_ref, &call_args);
    let results = ctx.builder.inst_results(call);
    if results.is_empty() {
        return None;
    }
    let raw_val = results[0];

    // If caller expects Any but we returned a concrete native type, wrap as TokValue
    if matches!(result_ty, Type::Any) && !matches!(spec_ret_type, Type::Any) {
        let (tag, data) = to_tokvalue(ctx, raw_val, &spec_ret_type);
        Some(alloc_tokvalue_on_stack(ctx, tag, data))
    } else {
        Some(coerce_value(ctx, raw_val, &spec_ret_type, result_ty))
    }
}

// ─── Inline filter/reduce ──────────────────────────────────────────────

/// Check if a filter/reduce lambda can be inlined at compile time.
fn can_inline_hof(lambda_expr: &HirExpr, arr_ty: &Type, expected_params: usize) -> bool {
    if let HirExprKind::Lambda { params, .. } = &lambda_expr.kind {
        if params.len() != expected_params {
            return false;
        }
        // Array element type must be concrete (not Any) for native-type inlining to help
        match arr_ty {
            Type::Array(inner) => !matches!(inner.as_ref(), Type::Any),
            _ => false,
        }
    } else {
        false
    }
}

/// Compile `arr?>\(x)=pred` as an inline loop instead of a runtime call.
fn compile_inline_filter(
    ctx: &mut FuncCtx,
    arr_expr: &HirExpr,
    lambda_expr: &HirExpr,
    result_ty: &Type,
) -> Option<Value> {
    let HirExprKind::Lambda { params, body, .. } = &lambda_expr.kind else { unreachable!() };
    let elem_type = match &arr_expr.ty {
        Type::Array(inner) => inner.as_ref().clone(),
        _ => Type::Any,
    };
    let param_name = &params[0].name;

    // Compile source array
    let arr_raw = compile_expr(ctx, arr_expr).unwrap();
    let arr = if matches!(&arr_expr.ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
        ctx.builder.ins().load(types::I64, MemFlags::trusted(), arr_raw, 8)
    } else {
        arr_raw
    };

    // Allocate result array
    let alloc_ref = ctx.get_runtime_func_ref("tok_array_alloc");
    let alloc_call = ctx.builder.ins().call(alloc_ref, &[]);
    let result_arr = ctx.builder.inst_results(alloc_call)[0];

    // Get source length
    let len_ref = ctx.get_runtime_func_ref("tok_array_len");
    let len_call = ctx.builder.ins().call(len_ref, &[arr]);
    let len_val = ctx.builder.inst_results(len_call)[0];

    // Loop index
    let idx_var = ctx.new_var(types::I64);
    let zero = ctx.builder.ins().iconst(types::I64, 0);
    ctx.builder.def_var(idx_var, zero);

    // Element variable bound to lambda param
    let ct = cl_type_or_i64(&elem_type);
    let elem_var = ctx.new_var(ct);
    let elem_zero = zero_value(&mut ctx.builder, ct);
    ctx.builder.def_var(elem_var, elem_zero);

    // Save old binding and insert lambda param
    let old_binding = ctx.vars.remove(param_name);
    ctx.vars.insert(param_name.clone(), (elem_var, elem_type.clone()));

    // Loop blocks
    let header_block = ctx.builder.create_block();
    let body_block = ctx.builder.create_block();
    let push_block = ctx.builder.create_block();
    let inc_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();

    ctx.builder.ins().jump(header_block, &[]);
    ctx.builder.switch_to_block(header_block);

    // Condition: i < len
    let current_idx = ctx.builder.use_var(idx_var);
    let cond = ctx.builder.ins().icmp(
        cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
        current_idx,
        len_val,
    );
    ctx.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

    ctx.builder.switch_to_block(body_block);
    ctx.builder.seal_block(body_block);

    // Get element as (tag, data)
    let current_idx = ctx.builder.use_var(idx_var);
    let get_ref = ctx.get_runtime_func_ref("tok_array_get");
    let get_call = ctx.builder.ins().call(get_ref, &[arr, current_idx]);
    let get_results = ctx.builder.inst_results(get_call);
    let elem_tag = get_results[0];
    let elem_data = get_results[1];

    // Extract native value for the lambda body
    let elem_native = from_tokvalue(ctx, elem_tag, elem_data, &elem_type);
    ctx.builder.def_var(elem_var, elem_native);

    // Retype and compile lambda body inline
    let mut type_map = HashMap::new();
    type_map.insert(param_name.clone(), elem_type.clone());
    let retyped = retype_body(body, &type_map);
    // Unwrap trailing Return(Some(expr)) → Expr(expr) since we're inlining
    let retyped = unwrap_return_stmts(retyped);
    let pred_result = compile_body(ctx, &retyped, &Type::Bool);

    if let Some(pred_val) = pred_result {
        if !ctx.block_terminated {
            // Determine predicate result type
            let pred_ty = retyped.last()
                .and_then(|s| match s { HirStmt::Expr(e) => Some(e.ty.clone()), _ => None })
                .unwrap_or(Type::Bool);
            let bool_val = to_bool(ctx, pred_val, &pred_ty);

            ctx.builder.ins().brif(bool_val, push_block, &[], inc_block, &[]);
        }

        // Push block: add element to result array
        ctx.builder.switch_to_block(push_block);
        ctx.builder.seal_block(push_block);
        let push_ref = ctx.get_runtime_func_ref("tok_array_push");
        ctx.builder.ins().call(push_ref, &[result_arr, elem_tag, elem_data]);
        ctx.builder.ins().jump(inc_block, &[]);
    } else {
        if !ctx.block_terminated {
            ctx.builder.ins().jump(inc_block, &[]);
        }
    }

    // Increment
    ctx.block_terminated = false;
    ctx.builder.switch_to_block(inc_block);
    ctx.builder.seal_block(inc_block);
    let current_idx = ctx.builder.use_var(idx_var);
    let one = ctx.builder.ins().iconst(types::I64, 1);
    let next_idx = ctx.builder.ins().iadd(current_idx, one);
    ctx.builder.def_var(idx_var, next_idx);
    ctx.builder.ins().jump(header_block, &[]);

    ctx.builder.seal_block(header_block);
    ctx.builder.switch_to_block(exit_block);
    ctx.builder.seal_block(exit_block);
    ctx.block_terminated = false;

    // Restore old binding
    ctx.vars.remove(param_name);
    if let Some(old) = old_binding {
        ctx.vars.insert(param_name.clone(), old);
    }

    // If caller expects Any, wrap result
    if matches!(result_ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
        let tag = ctx.builder.ins().iconst(types::I64, TAG_ARRAY as i64);
        return Some(alloc_tokvalue_on_stack(ctx, tag, result_arr));
    }
    Some(result_arr)
}

/// Compile `arr/>init \(acc x)=body` as an inline loop instead of a runtime call.
fn compile_inline_reduce(
    ctx: &mut FuncCtx,
    arr_expr: &HirExpr,
    init_expr: &HirExpr,
    lambda_expr: &HirExpr,
    result_ty: &Type,
) -> Option<Value> {
    let HirExprKind::Lambda { params, body, .. } = &lambda_expr.kind else { unreachable!() };
    let elem_type = match &arr_expr.ty {
        Type::Array(inner) => inner.as_ref().clone(),
        _ => Type::Any,
    };
    let acc_name = &params[0].name;
    let elem_name = &params[1].name;

    // Compile source array
    let arr_raw = compile_expr(ctx, arr_expr).unwrap();
    let arr = if matches!(&arr_expr.ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
        ctx.builder.ins().load(types::I64, MemFlags::trusted(), arr_raw, 8)
    } else {
        arr_raw
    };

    // Get length
    let len_ref = ctx.get_runtime_func_ref("tok_array_len");
    let len_call = ctx.builder.ins().call(len_ref, &[arr]);
    let len_val = ctx.builder.inst_results(len_call)[0];

    // Determine accumulator type from init expression
    let acc_type = if matches!(&init_expr.kind, HirExprKind::Nil) {
        // No explicit init — acc type is same as element type
        elem_type.clone()
    } else {
        init_expr.ty.clone()
    };
    let acc_ct = cl_type_or_i64(&acc_type);

    // Compile init value and determine start index
    let (init_val, start_idx) = if matches!(&init_expr.kind, HirExprKind::Nil) {
        // No init: use first element, start from 1
        let zero_idx = ctx.builder.ins().iconst(types::I64, 0);
        let get_ref = ctx.get_runtime_func_ref("tok_array_get");
        let get_call = ctx.builder.ins().call(get_ref, &[arr, zero_idx]);
        let results = ctx.builder.inst_results(get_call);
        let first = from_tokvalue(ctx, results[0], results[1], &elem_type);
        let one = ctx.builder.ins().iconst(types::I64, 1);
        (first, one)
    } else {
        let iv = compile_expr(ctx, init_expr).unwrap();
        let zero = ctx.builder.ins().iconst(types::I64, 0);
        (iv, zero)
    };

    // Accumulator variable
    let acc_var = ctx.new_var(acc_ct);
    ctx.builder.def_var(acc_var, init_val);

    // Loop index
    let idx_var = ctx.new_var(types::I64);
    ctx.builder.def_var(idx_var, start_idx);

    // Element variable
    let elem_ct = cl_type_or_i64(&elem_type);
    let elem_var = ctx.new_var(elem_ct);
    let elem_zero = zero_value(&mut ctx.builder, elem_ct);
    ctx.builder.def_var(elem_var, elem_zero);

    // Bind lambda params
    let old_acc_binding = ctx.vars.remove(acc_name);
    let old_elem_binding = ctx.vars.remove(elem_name);
    ctx.vars.insert(acc_name.clone(), (acc_var, acc_type.clone()));
    ctx.vars.insert(elem_name.clone(), (elem_var, elem_type.clone()));

    // Loop blocks
    let header_block = ctx.builder.create_block();
    let body_block = ctx.builder.create_block();
    let inc_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();

    ctx.builder.ins().jump(header_block, &[]);
    ctx.builder.switch_to_block(header_block);

    // Condition: i < len
    let current_idx = ctx.builder.use_var(idx_var);
    let cond = ctx.builder.ins().icmp(
        cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
        current_idx,
        len_val,
    );
    ctx.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

    ctx.builder.switch_to_block(body_block);
    ctx.builder.seal_block(body_block);

    // Get element
    let current_idx = ctx.builder.use_var(idx_var);
    let get_ref = ctx.get_runtime_func_ref("tok_array_get");
    let get_call = ctx.builder.ins().call(get_ref, &[arr, current_idx]);
    let get_results = ctx.builder.inst_results(get_call);
    let elem_native = from_tokvalue(ctx, get_results[0], get_results[1], &elem_type);
    ctx.builder.def_var(elem_var, elem_native);

    // Retype and compile lambda body inline
    let mut type_map = HashMap::new();
    type_map.insert(acc_name.clone(), acc_type.clone());
    type_map.insert(elem_name.clone(), elem_type.clone());
    let retyped = retype_body(body, &type_map);
    // Unwrap trailing Return(Some(expr)) → Expr(expr) since we're inlining
    let retyped = unwrap_return_stmts(retyped);
    let body_result = compile_body(ctx, &retyped, &acc_type);

    if let Some(val) = body_result {
        // Determine body result type
        let body_ty = retyped.last()
            .and_then(|s| match s { HirStmt::Expr(e) => Some(e.ty.clone()), _ => None })
            .unwrap_or(acc_type.clone());
        let coerced = coerce_value(ctx, val, &body_ty, &acc_type);
        ctx.builder.def_var(acc_var, coerced);
    }

    // Increment
    if !ctx.block_terminated {
        ctx.builder.ins().jump(inc_block, &[]);
    }
    ctx.block_terminated = false;

    ctx.builder.switch_to_block(inc_block);
    ctx.builder.seal_block(inc_block);
    let current_idx = ctx.builder.use_var(idx_var);
    let one = ctx.builder.ins().iconst(types::I64, 1);
    let next_idx = ctx.builder.ins().iadd(current_idx, one);
    ctx.builder.def_var(idx_var, next_idx);
    ctx.builder.ins().jump(header_block, &[]);

    ctx.builder.seal_block(header_block);
    ctx.builder.switch_to_block(exit_block);
    ctx.builder.seal_block(exit_block);
    ctx.block_terminated = false;

    // Restore bindings
    ctx.vars.remove(acc_name);
    ctx.vars.remove(elem_name);
    if let Some(old) = old_acc_binding {
        ctx.vars.insert(acc_name.clone(), old);
    }
    if let Some(old) = old_elem_binding {
        ctx.vars.insert(elem_name.clone(), old);
    }

    // Return accumulator
    let final_acc = ctx.builder.use_var(acc_var);
    if matches!(result_ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
        let (tag, data) = to_tokvalue(ctx, final_acc, &acc_type);
        Some(alloc_tokvalue_on_stack(ctx, tag, data))
    } else {
        Some(coerce_value(ctx, final_acc, &acc_type, result_ty))
    }
}

fn compile_print_call(ctx: &mut FuncCtx, args: &[HirExpr], newline: bool) -> Option<Value> {
    for (i, arg) in args.iter().enumerate() {
        let val = compile_expr(ctx, arg).unwrap_or_else(|| {
            ctx.builder.ins().iconst(types::I64, 0)
        });
        let use_newline = newline && i == args.len() - 1;
        let func_name = match &arg.ty {
            Type::Int => {
                if use_newline { "tok_println_int" } else { "tok_print_int" }
            }
            Type::Float => {
                if use_newline { "tok_println_float" } else { "tok_print_float" }
            }
            Type::Str => {
                if use_newline { "tok_println_string" } else { "tok_print_string" }
            }
            Type::Bool => {
                if use_newline { "tok_println_bool" } else { "tok_print_bool" }
            }
            _ => {
                // Pack as TokValue
                let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                let func_name = if use_newline { "tok_println" } else { "tok_print" };
                let func_ref = ctx.get_runtime_func_ref(func_name);
                ctx.builder.ins().call(func_ref, &[tag, data]);
                continue;
            }
        };
        let func_ref = ctx.get_runtime_func_ref(func_name);
        ctx.builder.ins().call(func_ref, &[val]);
    }
    None
}

// ─── If/else ──────────────────────────────────────────────────────────

fn compile_if(
    ctx: &mut FuncCtx,
    cond: &HirExpr,
    then_body: &[HirStmt],
    then_expr: &Option<Box<HirExpr>>,
    else_body: &[HirStmt],
    else_expr: &Option<Box<HirExpr>>,
    result_ty: &Type,
) -> Option<Value> {
    let cond_val = compile_expr(ctx, cond).unwrap();
    let cond_bool = to_bool(ctx, cond_val, &cond.ty);

    let then_block = ctx.builder.create_block();
    let else_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();

    // If either branch is Any but result_ty is concrete, we must use Any semantics
    // internally because the Any branch might hold a different runtime type.
    let then_ty = then_expr.as_ref().map(|e| &e.ty);
    let else_ty = else_expr.as_ref().map(|e| &e.ty);
    let any_branch = then_ty.map_or(false, |t| matches!(t, Type::Any))
        || else_ty.map_or(false, |t| matches!(t, Type::Any));
    let needs_any_upgrade = any_branch && !matches!(result_ty, Type::Any | Type::Nil | Type::Never);
    let merge_ty = if needs_any_upgrade { &Type::Any } else { result_ty };

    // Does this if produce a value?
    let has_value = cl_type(merge_ty).is_some() && (then_expr.is_some() || else_expr.is_some());
    let result_cl_type = cl_type_or_i64(merge_ty);
    if has_value {
        ctx.builder.append_block_param(merge_block, result_cl_type);
    }

    ctx.builder.ins().brif(cond_bool, then_block, &[], else_block, &[]);

    // Then branch
    ctx.builder.switch_to_block(then_block);
    ctx.builder.seal_block(then_block);
    ctx.block_terminated = false;
    compile_body(ctx, then_body, &Type::Nil);
    let then_val = if !ctx.block_terminated {
        if let Some(te) = then_expr {
            compile_expr(ctx, te)
        } else {
            None
        }
    } else {
        None
    };
    let then_terminated = ctx.block_terminated;
    if !then_terminated {
        if has_value {
            let v = then_val.unwrap_or_else(|| zero_value(&mut ctx.builder, result_cl_type));
            let then_expr_ty = then_expr.as_ref().map(|e| &e.ty);
            let v = coerce_if_branch(ctx, v, then_expr_ty, merge_ty, result_cl_type);
            ctx.builder.ins().jump(merge_block, &[v]);
        } else {
            ctx.builder.ins().jump(merge_block, &[]);
        }
    } else {
        // Block was terminated (return/break) — fill the dead block with a trap
        // so Cranelift doesn't complain about unfilled blocks
        ctx.builder.ins().trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
    }

    // Else branch
    ctx.builder.switch_to_block(else_block);
    ctx.builder.seal_block(else_block);
    ctx.block_terminated = false;
    compile_body(ctx, else_body, &Type::Nil);
    let else_val = if !ctx.block_terminated {
        if let Some(ee) = else_expr {
            compile_expr(ctx, ee)
        } else {
            None
        }
    } else {
        None
    };
    let else_terminated = ctx.block_terminated;
    if !else_terminated {
        if has_value {
            let v = else_val.unwrap_or_else(|| zero_value(&mut ctx.builder, result_cl_type));
            let else_expr_ty = else_expr.as_ref().map(|e| &e.ty);
            let v = coerce_if_branch(ctx, v, else_expr_ty, merge_ty, result_cl_type);
            ctx.builder.ins().jump(merge_block, &[v]);
        } else {
            ctx.builder.ins().jump(merge_block, &[]);
        }
    } else {
        ctx.builder.ins().trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
    }

    // If both branches terminated (return/break), the merge block is unreachable
    // but we still need to switch to it for subsequent code
    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
    ctx.block_terminated = then_terminated && else_terminated;

    if has_value {
        let merge_val = ctx.builder.block_params(merge_block)[0];
        // If we upgraded to Any internally, coerce back to the original result type
        if needs_any_upgrade {
            Some(coerce_value(ctx, merge_val, &Type::Any, result_ty))
        } else {
            Some(merge_val)
        }
    } else {
        None
    }
}

// ─── Loops ────────────────────────────────────────────────────────────

fn compile_loop(
    ctx: &mut FuncCtx,
    kind: &HirLoopKind,
    body: &[HirStmt],
) {
    match kind {
        HirLoopKind::While(cond) => {
            let header_block = ctx.builder.create_block();
            let body_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            ctx.builder.ins().jump(header_block, &[]);
            ctx.builder.switch_to_block(header_block);

            let cond_val = compile_expr(ctx, cond).unwrap();
            let cond_bool = to_bool(ctx, cond_val, &cond.ty);
            ctx.builder.ins().brif(cond_bool, body_block, &[], exit_block, &[]);

            ctx.builder.switch_to_block(body_block);
            ctx.builder.seal_block(body_block);

            ctx.loop_stack.push((header_block, exit_block));
            compile_body(ctx, body, &Type::Nil);
            ctx.loop_stack.pop();

            ctx.builder.ins().jump(header_block, &[]);

            ctx.builder.seal_block(header_block);
            ctx.builder.switch_to_block(exit_block);
            ctx.builder.seal_block(exit_block);
        }

        HirLoopKind::ForRange {
            var,
            start,
            end,
            inclusive,
        } => {
            let start_val = compile_expr(ctx, start).unwrap();
            let end_val = compile_expr(ctx, end).unwrap();

            // Create loop variable
            let loop_var = ctx.new_var(types::I64);
            ctx.builder.def_var(loop_var, start_val);
            ctx.vars.insert(var.clone(), (loop_var, Type::Int));

            // Loop rotation: guard check → body → inc+check (back-edge is conditional)
            // This eliminates one unconditional jump per iteration vs condition-at-top.
            // Note: Cranelift may rematerialize end_val inside the loop for large constants
            // (>12-bit immediate), but this is harmless on ARM64's wide execution pipeline
            // and the rotation itself saves a branch per iteration.
            let body_block = ctx.builder.create_block();
            let inc_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            // Guard: skip loop entirely if start >= end
            let cc = if *inclusive {
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThanOrEqual
            } else {
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan
            };
            let guard_cond = ctx.builder.ins().icmp(cc, start_val, end_val);
            ctx.builder.ins().brif(guard_cond, body_block, &[], exit_block, &[]);

            // Body block
            ctx.builder.switch_to_block(body_block);

            ctx.loop_stack.push((inc_block, exit_block));
            compile_body(ctx, body, &Type::Nil);
            ctx.loop_stack.pop();

            if !ctx.block_terminated {
                ctx.builder.ins().jump(inc_block, &[]);
            }

            // Increment + back-edge condition check (replaces separate header block)
            ctx.builder.switch_to_block(inc_block);
            ctx.builder.seal_block(inc_block);
            let current = ctx.builder.use_var(loop_var);
            let one = ctx.builder.ins().iconst(types::I64, 1);
            let next = ctx.builder.ins().iadd(current, one);
            ctx.builder.def_var(loop_var, next);
            let cond = ctx.builder.ins().icmp(cc, next, end_val);
            ctx.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

            ctx.builder.seal_block(body_block);
            ctx.builder.switch_to_block(exit_block);
            ctx.builder.seal_block(exit_block);
            ctx.block_terminated = false;
        }

        HirLoopKind::ForEach { var, iter } => {
            let iter_val = compile_expr(ctx, iter).unwrap();

            // For Any-typed iterables, extract the actual pointer and use runtime dispatch
            let is_any_iter = matches!(&iter.ty, Type::Any | Type::Optional(_) | Type::Result(_));

            // Get length
            let len_val = if is_any_iter {
                let (tag, data) = to_tokvalue(ctx, iter_val, &iter.ty);
                let len_ref = ctx.get_runtime_func_ref("tok_value_len");
                let len_call = ctx.builder.ins().call(len_ref, &[tag, data]);
                ctx.builder.inst_results(len_call)[0]
            } else {
                let len_func = match &iter.ty {
                    Type::Array(_) => "tok_array_len",
                    Type::Str => "tok_string_len",
                    _ => "tok_array_len",
                };
                let len_ref = ctx.get_runtime_func_ref(len_func);
                let len_call = ctx.builder.ins().call(len_ref, &[iter_val]);
                ctx.builder.inst_results(len_call)[0]
            };

            // Index variable
            let idx_var = ctx.new_var(types::I64);
            let zero = ctx.builder.ins().iconst(types::I64, 0);
            ctx.builder.def_var(idx_var, zero);

            // Element variable — for Any iterables, elements are Any too
            let elem_type = match &iter.ty {
                Type::Array(inner) => inner.as_ref().clone(),
                Type::Str => Type::Str,
                _ => Type::Any,
            };
            let ct = cl_type_or_i64(&elem_type);
            let elem_var = ctx.new_var(ct);
            let elem_zero = zero_value(&mut ctx.builder, ct);
            ctx.builder.def_var(elem_var, elem_zero);
            ctx.vars.insert(var.clone(), (elem_var, elem_type.clone()));

            let header_block = ctx.builder.create_block();
            let body_block = ctx.builder.create_block();
            let inc_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            ctx.builder.ins().jump(header_block, &[]);
            ctx.builder.switch_to_block(header_block);

            let current_idx = ctx.builder.use_var(idx_var);
            let cond = ctx.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                current_idx,
                len_val,
            );
            ctx.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

            ctx.builder.switch_to_block(body_block);
            ctx.builder.seal_block(body_block);

            // Get element
            let current_idx = ctx.builder.use_var(idx_var);
            if is_any_iter {
                // For Any, use tok_value_index which dispatches by tag
                let (t_tag, t_data) = to_tokvalue(ctx, iter_val, &iter.ty);
                let idx_ref = ctx.get_runtime_func_ref("tok_value_index");
                let idx_call = ctx.builder.ins().call(idx_ref, &[t_tag, t_data, current_idx]);
                let results = ctx.builder.inst_results(idx_call);
                let elem = from_tokvalue(ctx, results[0], results[1], &elem_type);
                ctx.builder.def_var(elem_var, elem);
            } else {
                match &iter.ty {
                    Type::Array(_) => {
                        let get_ref = ctx.get_runtime_func_ref("tok_array_get");
                        let get_call = ctx.builder.ins().call(get_ref, &[iter_val, current_idx]);
                        let results = ctx.builder.inst_results(get_call);
                        let elem = from_tokvalue(ctx, results[0], results[1], &elem_type);
                        ctx.builder.def_var(elem_var, elem);
                    }
                    Type::Str => {
                        let get_ref = ctx.get_runtime_func_ref("tok_string_index");
                        let get_call = ctx.builder.ins().call(get_ref, &[iter_val, current_idx]);
                        let elem = ctx.builder.inst_results(get_call)[0];
                        ctx.builder.def_var(elem_var, elem);
                    }
                    _ => {
                        let get_ref = ctx.get_runtime_func_ref("tok_array_get");
                        let get_call = ctx.builder.ins().call(get_ref, &[iter_val, current_idx]);
                        let results = ctx.builder.inst_results(get_call);
                        let elem = from_tokvalue(ctx, results[0], results[1], &elem_type);
                        ctx.builder.def_var(elem_var, elem);
                    }
                }
            }

            ctx.loop_stack.push((inc_block, exit_block));
            compile_body(ctx, body, &Type::Nil);
            ctx.loop_stack.pop();

            ctx.builder.ins().jump(inc_block, &[]);

            ctx.builder.switch_to_block(inc_block);
            ctx.builder.seal_block(inc_block);
            let current_idx = ctx.builder.use_var(idx_var);
            let one = ctx.builder.ins().iconst(types::I64, 1);
            let next_idx = ctx.builder.ins().iadd(current_idx, one);
            ctx.builder.def_var(idx_var, next_idx);
            ctx.builder.ins().jump(header_block, &[]);

            ctx.builder.seal_block(header_block);
            ctx.builder.switch_to_block(exit_block);
            ctx.builder.seal_block(exit_block);
        }

        HirLoopKind::ForEachIndexed {
            idx_var: idx_name,
            val_var: val_name,
            iter,
        } => {
            // Similar to ForEach but also binds an index variable
            let iter_val = compile_expr(ctx, iter).unwrap();
            let len_ref = ctx.get_runtime_func_ref("tok_array_len");
            let len_call = ctx.builder.ins().call(len_ref, &[iter_val]);
            let len_val = ctx.builder.inst_results(len_call)[0];

            let idx_var = ctx.new_var(types::I64);
            let zero = ctx.builder.ins().iconst(types::I64, 0);
            ctx.builder.def_var(idx_var, zero);
            ctx.vars.insert(idx_name.clone(), (idx_var, Type::Int));

            let elem_type = match &iter.ty {
                Type::Array(inner) => inner.as_ref().clone(),
                _ => Type::Any,
            };
            let ct = cl_type_or_i64(&elem_type);
            let elem_var = ctx.new_var(ct);
            let elem_zero = zero_value(&mut ctx.builder, ct);
            ctx.builder.def_var(elem_var, elem_zero);
            ctx.vars.insert(val_name.clone(), (elem_var, elem_type.clone()));

            let header_block = ctx.builder.create_block();
            let body_block = ctx.builder.create_block();
            let inc_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            ctx.builder.ins().jump(header_block, &[]);
            ctx.builder.switch_to_block(header_block);

            let current_idx = ctx.builder.use_var(idx_var);
            let cond = ctx.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                current_idx,
                len_val,
            );
            ctx.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

            ctx.builder.switch_to_block(body_block);
            ctx.builder.seal_block(body_block);

            let current_idx = ctx.builder.use_var(idx_var);
            let get_ref = ctx.get_runtime_func_ref("tok_array_get");
            let get_call = ctx.builder.ins().call(get_ref, &[iter_val, current_idx]);
            let results = ctx.builder.inst_results(get_call);
            let elem = from_tokvalue(ctx, results[0], results[1], &elem_type);
            ctx.builder.def_var(elem_var, elem);

            ctx.loop_stack.push((inc_block, exit_block));
            compile_body(ctx, body, &Type::Nil);
            ctx.loop_stack.pop();

            ctx.builder.ins().jump(inc_block, &[]);

            ctx.builder.switch_to_block(inc_block);
            ctx.builder.seal_block(inc_block);
            let current_idx = ctx.builder.use_var(idx_var);
            let one = ctx.builder.ins().iconst(types::I64, 1);
            let next_idx = ctx.builder.ins().iadd(current_idx, one);
            ctx.builder.def_var(idx_var, next_idx);
            ctx.builder.ins().jump(header_block, &[]);

            ctx.builder.seal_block(header_block);
            ctx.builder.switch_to_block(exit_block);
            ctx.builder.seal_block(exit_block);
        }

        HirLoopKind::Infinite => {
            let body_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            ctx.builder.ins().jump(body_block, &[]);
            ctx.builder.switch_to_block(body_block);

            ctx.loop_stack.push((body_block, exit_block));
            compile_body(ctx, body, &Type::Nil);
            ctx.loop_stack.pop();

            ctx.builder.ins().jump(body_block, &[]);

            ctx.builder.seal_block(body_block);
            ctx.builder.switch_to_block(exit_block);
            ctx.builder.seal_block(exit_block);
        }
    }
}

/// Allocate a TokValue on the stack and store tag+data, returning a pointer.
fn alloc_tokvalue_on_stack(ctx: &mut FuncCtx, tag: Value, data: Value) -> Value {
    let ss = ctx.builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        16,
        3, // 8-byte alignment
    ));
    let addr = ctx.builder.ins().stack_addr(PTR, ss, 0);
    ctx.builder.ins().store(MemFlags::trusted(), tag, addr, 0);
    ctx.builder.ins().store(MemFlags::trusted(), data, addr, 8);
    addr
}

/// Coerce a value from one type to another.
/// E.g., Int → Any: wrap in stack-allocated TokValue. Any → Int: load and extract.
fn coerce_value(ctx: &mut FuncCtx, val: Value, from: &Type, to: &Type) -> Value {
    // Same type or both non-Any → no coercion needed
    if std::mem::discriminant(from) == std::mem::discriminant(to) {
        return val;
    }
    // Concrete → Any: wrap in stack TokValue
    if matches!(to, Type::Any) && !matches!(from, Type::Any) {
        let (tag, data) = to_tokvalue(ctx, val, from);
        return from_tokvalue(ctx, tag, data, &Type::Any);
    }
    // Any → Concrete: unwrap from stack TokValue
    if matches!(from, Type::Any) && !matches!(to, Type::Any) {
        let (tag, data) = to_tokvalue(ctx, val, from);
        return from_tokvalue(ctx, tag, data, to);
    }
    // Int → Float
    if matches!(from, Type::Int) && matches!(to, Type::Float) {
        return ctx.builder.ins().fcvt_from_sint(types::F64, val);
    }
    // Float → Int
    if matches!(from, Type::Float) && matches!(to, Type::Int) {
        return ctx.builder.ins().fcvt_to_sint_sat(types::I64, val);
    }
    val
}

/// If the expr type is Any, extract the raw pointer (data field at offset 8) from the
/// TokValue. Otherwise return the value as-is (it's already a raw pointer).
/// Used for builtins that expect a concrete pointer (Array, Map, String, etc.)
/// but may receive an Any-typed TokValue pointer.
fn unwrap_any_ptr(ctx: &mut FuncCtx, val: Value, ty: &Type) -> Value {
    if matches!(ty, Type::Any) {
        ctx.builder.ins().load(types::I64, MemFlags::trusted(), val, 8)
    } else {
        val
    }
}

// ─── TokValue packing/unpacking ───────────────────────────────────────

/// TAG constants matching the runtime.
const TAG_NIL: i64 = 0;
const TAG_INT: i64 = 1;
const TAG_FLOAT: i64 = 2;
const TAG_BOOL: i64 = 3;
const TAG_STRING: i64 = 4;
const TAG_ARRAY: i64 = 5;
const TAG_MAP: i64 = 6;
const TAG_TUPLE: i64 = 7;
const TAG_FUNC: i64 = 8;
const TAG_CHANNEL: i64 = 9;
const TAG_HANDLE: i64 = 10;

/// Pack a typed value into (tag: I64, data: I64) for runtime calls that take TokValue.
fn to_tokvalue(ctx: &mut FuncCtx, val: Value, ty: &Type) -> (Value, Value) {
    let (tag, data) = match ty {
        Type::Int => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_INT);
            (tag, val) // i64 fits in data word
        }
        Type::Float => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_FLOAT);
            let bits = ctx.builder.ins().bitcast(types::I64, MemFlags::new(), val);
            (tag, bits)
        }
        Type::Bool => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_BOOL);
            let ext = ctx.builder.ins().uextend(types::I64, val);
            (tag, ext)
        }
        Type::Str => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_STRING);
            (tag, val)
        }
        Type::Array(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_ARRAY);
            (tag, val)
        }
        Type::Map(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_MAP);
            (tag, val)
        }
        Type::Tuple(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_TUPLE);
            (tag, val)
        }
        Type::Func(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_FUNC);
            (tag, val)
        }
        Type::Channel(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_CHANNEL);
            (tag, val)
        }
        Type::Handle(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_HANDLE);
            (tag, val)
        }
        Type::Nil | Type::Never => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_NIL);
            let data = ctx.builder.ins().iconst(types::I64, 0);
            (tag, data)
        }
        Type::Any | Type::Optional(_) | Type::Result(_) | Type::Range => {
            // `Any` values are stored as stack-allocated TokValues (ptr to 16-byte struct).
            // Load tag (offset 0) and data (offset 8) from the stack slot.
            let tag = ctx.builder.ins().load(types::I64, MemFlags::trusted(), val, 0);
            let data = ctx.builder.ins().load(types::I64, MemFlags::trusted(), val, 8);
            return (tag, data);
        }
    };
    (tag, data)
}

/// Unpack a (tag: I64, data: I64) TokValue into the expected typed value.
fn from_tokvalue(ctx: &mut FuncCtx, tag: Value, data: Value, ty: &Type) -> Value {
    match ty {
        Type::Int => data,
        Type::Float => ctx.builder.ins().bitcast(types::F64, MemFlags::new(), data),
        Type::Bool => ctx.builder.ins().ireduce(types::I8, data),
        Type::Str | Type::Array(_) | Type::Map(_) | Type::Tuple(_) | Type::Func(_)
        | Type::Channel(_) | Type::Handle(_) => data, // pointer
        Type::Any | Type::Optional(_) | Type::Result(_) | Type::Range => {
            alloc_tokvalue_on_stack(ctx, tag, data)
        }
        _ => data,
    }
}

/// Convert a value to a boolean (i8) for branching.
fn to_bool(ctx: &mut FuncCtx, val: Value, ty: &Type) -> Value {
    match ty {
        Type::Bool => val,
        Type::Int => {
            let zero = ctx.builder.ins().iconst(types::I64, 0);
            ctx.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                val,
                zero,
            )
        }
        Type::Float => {
            let zero = ctx.builder.ins().f64const(0.0);
            ctx.builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::NotEqual,
                val,
                zero,
            )
        }
        Type::Str | Type::Array(_) | Type::Map(_) | Type::Tuple(_) => {
            // Non-null pointer = truthy
            let zero = ctx.builder.ins().iconst(PTR, 0);
            ctx.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                val,
                zero,
            )
        }
        _ => {
            // For Any types, use runtime truthiness
            let (tag, data) = to_tokvalue(ctx, val, ty);
            let func_ref = ctx.get_runtime_func_ref("tok_value_truthiness");
            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
            ctx.builder.inst_results(call)[0]
        }
    }
}

/// Create a zero/default value of the given Cranelift type.
/// Coerce an if-branch value to match the merge block's expected type.
/// Handles type mismatches between branch values and the merge block parameter.
fn coerce_if_branch(
    ctx: &mut FuncCtx,
    val: Value,
    branch_expr_ty: Option<&Type>,
    result_ty: &Type,
    result_cl_type: types::Type,
) -> Value {
    if let Some(branch_ty) = branch_expr_ty {
        // If merge block is Any: wrap concrete branch values as TokValue
        if matches!(result_ty, Type::Any) && !matches!(branch_ty, Type::Any) {
            let (tag, data) = to_tokvalue(ctx, val, branch_ty);
            return alloc_tokvalue_on_stack(ctx, tag, data);
        }
        // Both are Any: no coercion needed
        if matches!(result_ty, Type::Any) && matches!(branch_ty, Type::Any) {
            return val;
        }
    }
    coerce_cl_value(ctx, val, result_cl_type)
}

/// Coerce a Cranelift value to the target type by inserting uextend/bitcast as needed.
fn coerce_cl_value(ctx: &mut FuncCtx, val: Value, target: types::Type) -> Value {
    let actual = ctx.builder.func.dfg.value_type(val);
    if actual == target {
        return val;
    }
    // i8 → i64: uextend (Bool → Int/Any)
    if actual == types::I8 && target == types::I64 {
        return ctx.builder.ins().uextend(types::I64, val);
    }
    // f64 → i64: bitcast (Float → Any)
    if actual == types::F64 && target == types::I64 {
        return ctx.builder.ins().bitcast(types::I64, MemFlags::new(), val);
    }
    // i64 → f64: bitcast (Any → Float)
    if actual == types::I64 && target == types::F64 {
        return ctx.builder.ins().bitcast(types::F64, MemFlags::new(), val);
    }
    // i64 → i8: ireduce (Int → Bool)
    if actual == types::I64 && target == types::I8 {
        return ctx.builder.ins().ireduce(types::I8, val);
    }
    val
}

fn zero_value(builder: &mut FunctionBuilder, ty: types::Type) -> Value {
    if ty == types::F64 {
        builder.ins().f64const(0.0)
    } else if ty == types::F32 {
        builder.ins().f32const(0.0)
    } else {
        builder.ins().iconst(ty, 0)
    }
}

// ─── HIR type rewriting for specialized lambda bodies ──────────────────

/// Retype a lambda body by replacing `Any` types with concrete types from the given map.
/// This enables specialized lambda bodies to compile with native arithmetic instead of
/// going through the Any (TokValue) code paths.
/// Convert Return(Some(expr)) → Expr(expr) in a statement list.
/// Used when inlining a lambda body: the lambda's "return" should produce a value
/// in the current block, not jump to the enclosing function's return block.
fn unwrap_return_stmts(stmts: Vec<HirStmt>) -> Vec<HirStmt> {
    stmts.into_iter().map(|s| match s {
        HirStmt::Return(Some(expr)) => HirStmt::Expr(expr),
        HirStmt::Return(None) => HirStmt::Expr(HirExpr { kind: HirExprKind::Nil, ty: Type::Nil }),
        other => other,
    }).collect()
}

fn retype_body(body: &[HirStmt], type_map: &HashMap<String, Type>) -> Vec<HirStmt> {
    body.iter().map(|s| retype_stmt(s, type_map)).collect()
}

fn retype_stmt(stmt: &HirStmt, type_map: &HashMap<String, Type>) -> HirStmt {
    match stmt {
        HirStmt::Expr(e) => HirStmt::Expr(retype_expr(e, type_map)),
        HirStmt::Assign { name, ty, value } => {
            let new_value = retype_expr(value, type_map);
            HirStmt::Assign {
                name: name.clone(),
                ty: if matches!(ty, Type::Any) { new_value.ty.clone() } else { ty.clone() },
                value: new_value,
            }
        }
        HirStmt::Return(Some(e)) => HirStmt::Return(Some(retype_expr(e, type_map))),
        other => other.clone(),
    }
}

fn retype_expr(expr: &HirExpr, type_map: &HashMap<String, Type>) -> HirExpr {
    let mut e = expr.clone();
    match &mut e.kind {
        HirExprKind::Ident(name) => {
            if let Some(ty) = type_map.get(name.as_str()) {
                e.ty = ty.clone();
            }
        }
        HirExprKind::BinOp { left, right, op } => {
            *left = Box::new(retype_expr(left, type_map));
            *right = Box::new(retype_expr(right, type_map));
            // Propagate: infer result type from children
            e.ty = infer_binop_type(&left.ty, &right.ty, *op);
        }
        HirExprKind::UnaryOp { operand, op: _ } => {
            *operand = Box::new(retype_expr(operand, type_map));
            // Neg preserves type, Not → Bool
            e.ty = match &operand.ty {
                Type::Int => Type::Int,
                Type::Float => Type::Float,
                _ => e.ty.clone(),
            };
        }
        HirExprKind::Call { func, args } => {
            *func = Box::new(retype_expr(func, type_map));
            for arg in args.iter_mut() {
                *arg = retype_expr(arg, type_map);
            }
            // Don't change the call's result type — it depends on the callee
        }
        // Literals keep their types — no rewriting needed
        HirExprKind::Int(_) | HirExprKind::Float(_) | HirExprKind::Bool(_)
        | HirExprKind::Str(_) | HirExprKind::Nil => {}
        // For other complex nodes, just leave as-is
        _ => {}
    }
    e
}

/// Infer the result type of a binary operation given the types of both operands.
fn infer_binop_type(left: &Type, right: &Type, op: HirBinOp) -> Type {
    use HirBinOp::*;
    match op {
        // Comparison ops always return Bool
        Eq | Neq | Lt | Gt | LtEq | GtEq => Type::Bool,
        // Logical ops return Bool
        And | Or => Type::Bool,
        // Arithmetic: Int op Int → Int, Float involved → Float
        Add | Sub | Mul | Div | Mod | Pow | BitAnd | BitOr | BitXor | Shl | Shr => {
            match (left, right) {
                (Type::Int, Type::Int) => Type::Int,
                (Type::Float, Type::Float) => Type::Float,
                (Type::Int, Type::Float) | (Type::Float, Type::Int) => Type::Float,
                _ => Type::Any, // Can't determine — leave as Any
            }
        }
    }
}

// ─── Free variable analysis for closure captures ──────────────────────

/// Collect all free variables referenced in a lambda body that are not in `bound` (params + locals).
/// Returns the names of variables that need to be captured from the enclosing scope.
fn collect_free_vars(body: &[HirStmt], param_names: &HashSet<String>) -> HashSet<String> {
    let mut free = HashSet::new();
    let mut locals = param_names.clone();
    for stmt in body {
        collect_free_vars_stmt(stmt, &mut locals, &mut free);
    }
    free
}

fn collect_free_vars_stmt(stmt: &HirStmt, locals: &mut HashSet<String>, free: &mut HashSet<String>) {
    match stmt {
        HirStmt::Assign { name, value, .. } => {
            // The RHS may reference free vars (before the local is defined)
            collect_free_vars_expr(&value, locals, free);
            locals.insert(name.clone());
        }
        HirStmt::Expr(expr) => {
            collect_free_vars_expr(expr, locals, free);
        }
        HirStmt::Return(opt_expr) => {
            if let Some(expr) = opt_expr {
                collect_free_vars_expr(expr, locals, free);
            }
        }
        HirStmt::FuncDecl { name, .. } => {
            // The function name becomes a local, its body is a separate scope
            locals.insert(name.clone());
        }
        HirStmt::IndexAssign { target, index, value, .. } => {
            collect_free_vars_expr(target, locals, free);
            collect_free_vars_expr(index, locals, free);
            collect_free_vars_expr(value, locals, free);
        }
        HirStmt::MemberAssign { target, value, .. } => {
            collect_free_vars_expr(target, locals, free);
            collect_free_vars_expr(value, locals, free);
        }
        HirStmt::Import(_) => {}
        HirStmt::Break | HirStmt::Continue => {}
    }
}

fn collect_free_vars_expr(expr: &HirExpr, locals: &HashSet<String>, free: &mut HashSet<String>) {
    match &expr.kind {
        HirExprKind::Ident(name) => {
            if !locals.contains(name) {
                free.insert(name.clone());
            }
        }
        HirExprKind::Int(_) | HirExprKind::Float(_) | HirExprKind::Bool(_) | HirExprKind::Nil => {}
        HirExprKind::Str(_) => {}
        HirExprKind::Array(elems) => {
            for e in elems {
                collect_free_vars_expr(e, locals, free);
            }
        }
        HirExprKind::Map(entries) => {
            for (_k, v) in entries {
                collect_free_vars_expr(v, locals, free);
            }
        }
        HirExprKind::Tuple(elems) => {
            for e in elems {
                collect_free_vars_expr(e, locals, free);
            }
        }
        HirExprKind::BinOp { left, right, .. } => {
            collect_free_vars_expr(left, locals, free);
            collect_free_vars_expr(right, locals, free);
        }
        HirExprKind::UnaryOp { operand, .. } => {
            collect_free_vars_expr(operand, locals, free);
        }
        HirExprKind::Index { target, index } => {
            collect_free_vars_expr(target, locals, free);
            collect_free_vars_expr(index, locals, free);
        }
        HirExprKind::Member { target, .. } => {
            collect_free_vars_expr(target, locals, free);
        }
        HirExprKind::Call { func, args } => {
            collect_free_vars_expr(func, locals, free);
            for a in args {
                collect_free_vars_expr(a, locals, free);
            }
        }
        HirExprKind::RuntimeCall { args, .. } => {
            for a in args {
                collect_free_vars_expr(a, locals, free);
            }
        }
        HirExprKind::If { cond, then_body, then_expr, else_body, else_expr } => {
            collect_free_vars_expr(cond, locals, free);
            for s in then_body {
                collect_free_vars_stmt(s, &mut locals.clone(), free);
            }
            if let Some(e) = then_expr {
                collect_free_vars_expr(e, locals, free);
            }
            for s in else_body {
                collect_free_vars_stmt(s, &mut locals.clone(), free);
            }
            if let Some(e) = else_expr {
                collect_free_vars_expr(e, locals, free);
            }
        }
        HirExprKind::Loop { kind, body } => {
            // Collect free vars from the loop kind (range start/end, condition, iterator)
            let mut loop_locals = locals.clone();
            match kind.as_ref() {
                HirLoopKind::ForRange { var, start, end, .. } => {
                    collect_free_vars_expr(start, locals, free);
                    collect_free_vars_expr(end, locals, free);
                    loop_locals.insert(var.clone());
                }
                HirLoopKind::ForEach { var, iter } => {
                    collect_free_vars_expr(iter, locals, free);
                    loop_locals.insert(var.clone());
                }
                HirLoopKind::ForEachIndexed { idx_var, val_var, iter } => {
                    collect_free_vars_expr(iter, locals, free);
                    loop_locals.insert(idx_var.clone());
                    loop_locals.insert(val_var.clone());
                }
                HirLoopKind::While(cond) => {
                    collect_free_vars_expr(cond, locals, free);
                }
                HirLoopKind::Infinite => {}
            }
            for s in body {
                collect_free_vars_stmt(s, &mut loop_locals, free);
            }
        }
        HirExprKind::Lambda { params, body, .. } => {
            // Nested lambda: its params are bound, but it may capture from our scope
            let mut inner_locals = locals.clone();
            for p in params {
                inner_locals.insert(p.name.clone());
            }
            for s in body {
                collect_free_vars_stmt(s, &mut inner_locals, free);
            }
        }
        HirExprKind::Length(inner) => {
            collect_free_vars_expr(inner, locals, free);
        }
        HirExprKind::Block { stmts, expr } => {
            let mut block_locals = locals.clone();
            for s in stmts {
                collect_free_vars_stmt(s, &mut block_locals, free);
            }
            if let Some(e) = expr {
                collect_free_vars_expr(e, &block_locals, free);
            }
        }
        HirExprKind::Range { start, end, .. } => {
            collect_free_vars_expr(start, locals, free);
            collect_free_vars_expr(end, locals, free);
        }
        HirExprKind::Go(inner) => {
            collect_free_vars_expr(inner, locals, free);
        }
        HirExprKind::Receive(inner) => {
            collect_free_vars_expr(inner, locals, free);
        }
        HirExprKind::Send { chan, value } => {
            collect_free_vars_expr(chan, locals, free);
            collect_free_vars_expr(value, locals, free);
        }
        HirExprKind::Select(arms) => {
            for arm in arms {
                match arm {
                    HirSelectArm::Recv { chan, body, .. } => {
                        collect_free_vars_expr(chan, locals, free);
                        for s in body {
                            collect_free_vars_stmt(s, &mut locals.clone(), free);
                        }
                    }
                    HirSelectArm::Send { chan, value, body } => {
                        collect_free_vars_expr(chan, locals, free);
                        collect_free_vars_expr(value, locals, free);
                        for s in body {
                            collect_free_vars_stmt(s, &mut locals.clone(), free);
                        }
                    }
                    HirSelectArm::Default(body) => {
                        for s in body {
                            collect_free_vars_stmt(s, &mut locals.clone(), free);
                        }
                    }
                }
            }
        }
    }
}

