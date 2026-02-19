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
    AbiParam, Block, Function, InstBuilder, MemFlags, SigRef, StackSlotData, StackSlotKind,
    UserFuncName, Value,
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
        self.declare_runtime_func("tok_print", &[PTR, types::I64], &[]); // TokValue as 2 words

        // String
        self.declare_runtime_func("tok_string_alloc", &[PTR, types::I64], &[PTR]);
        self.declare_runtime_func("tok_string_concat", &[PTR, PTR], &[PTR]);
        self.declare_runtime_func("tok_string_len", &[PTR], &[types::I64]);
        self.declare_runtime_func("tok_string_eq", &[PTR, PTR], &[types::I8]);
        self.declare_runtime_func("tok_string_cmp", &[PTR, PTR], &[types::I64]);
        self.declare_runtime_func("tok_string_index", &[PTR, types::I64], &[PTR]);
        self.declare_runtime_func("tok_string_slice", &[PTR, types::I64, types::I64], &[PTR]);
        self.declare_runtime_func("tok_string_repeat", &[PTR, types::I64], &[PTR]);
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
        self.declare_runtime_func(
            "tok_array_reduce",
            &[PTR, types::I64, types::I64, PTR],
            &[types::I64, types::I64],
        );
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
        self.declare_runtime_func(
            "tok_value_index",
            &[PTR, types::I64, types::I64],
            &[PTR, types::I64],
        );

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
        self.declare_runtime_func(
            "tok_channel_try_send",
            &[PTR, PTR, types::I64],
            &[types::I8],
        );
        self.declare_runtime_func("tok_channel_try_recv", &[PTR, PTR], &[types::I8]);

        // Goroutine
        self.declare_runtime_func("tok_go", &[PTR, PTR], &[PTR]);
        self.declare_runtime_func("tok_handle_join", &[PTR], &[PTR, types::I64]);

        // Refcount
        self.declare_runtime_func("tok_rc_inc", &[PTR], &[]);
        self.declare_runtime_func("tok_rc_dec", &[PTR], &[types::I8]);
        self.declare_runtime_func("tok_value_rc_dec", &[types::I64, types::I64], &[]);
        self.declare_runtime_func("tok_string_free", &[PTR], &[]);

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
        self.declare_runtime_func(
            "tok_value_slice",
            &[PTR, types::I64, types::I64, types::I64],
            &[PTR, types::I64],
        );
        self.declare_runtime_func("tok_rand", &[], &[types::F64]);
        self.declare_runtime_func("tok_pow_f64", &[types::F64, types::F64], &[types::F64]);
        self.declare_runtime_func("tok_pow_int", &[types::I64, types::I64], &[types::I64]);

        // TokValue → concrete type extraction
        self.declare_runtime_func("tok_value_to_int", &[PTR, types::I64], &[types::I64]);
        self.declare_runtime_func("tok_value_to_float", &[PTR, types::I64], &[types::F64]);
        self.declare_runtime_func("tok_value_to_bool", &[PTR, types::I64], &[types::I8]);

        // Value ops (for Any type dispatch)
        self.declare_runtime_func(
            "tok_value_add",
            &[PTR, types::I64, PTR, types::I64],
            &[PTR, types::I64],
        );
        self.declare_runtime_func(
            "tok_value_sub",
            &[PTR, types::I64, PTR, types::I64],
            &[PTR, types::I64],
        );
        self.declare_runtime_func(
            "tok_value_mul",
            &[PTR, types::I64, PTR, types::I64],
            &[PTR, types::I64],
        );
        self.declare_runtime_func(
            "tok_value_div",
            &[PTR, types::I64, PTR, types::I64],
            &[PTR, types::I64],
        );
        self.declare_runtime_func(
            "tok_value_mod",
            &[PTR, types::I64, PTR, types::I64],
            &[PTR, types::I64],
        );
        self.declare_runtime_func(
            "tok_value_pow",
            &[PTR, types::I64, PTR, types::I64],
            &[PTR, types::I64],
        );
        self.declare_runtime_func("tok_value_negate", &[PTR, types::I64], &[PTR, types::I64]);
        self.declare_runtime_func(
            "tok_value_eq",
            &[PTR, types::I64, PTR, types::I64],
            &[types::I8],
        );
        self.declare_runtime_func(
            "tok_value_lt",
            &[PTR, types::I64, PTR, types::I64],
            &[types::I8],
        );
        self.declare_runtime_func("tok_value_truthiness", &[PTR, types::I64], &[types::I8]);
        self.declare_runtime_func("tok_value_not", &[PTR, types::I64], &[types::I8]);

        // Utility
        self.declare_runtime_func("tok_clock", &[], &[types::I64]);
        self.declare_runtime_func("tok_exit", &[types::I64], &[]);

        // New core builtins
        self.declare_runtime_func("tok_is", &[PTR, types::I64, PTR], &[types::I8]); // TokValue + string ptr -> bool
        self.declare_runtime_func("tok_array_pop", &[PTR], &[PTR, types::I64]); // arr -> TokValue (tuple)
        self.declare_runtime_func("tok_array_freq", &[PTR], &[PTR]); // arr -> map
        self.declare_runtime_func("tok_array_zip", &[PTR, PTR], &[PTR]); // arr, arr -> arr
        self.declare_runtime_func("tok_map_top", &[PTR, types::I64], &[PTR]); // map, n -> arr
        self.declare_runtime_func("tok_args", &[], &[PTR]); // -> arr
        self.declare_runtime_func("tok_env", &[PTR], &[PTR, types::I64]); // str -> TokValue

        // Stdlib module constructors — each returns *mut TokMap
        self.declare_runtime_func("tok_stdlib_math", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_str", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_os", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_io", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_json", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_llm", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_csv", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_fs", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_http", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_re", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_time", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_tmpl", &[], &[PTR]);
        self.declare_runtime_func("tok_stdlib_toon", &[], &[PTR]);

        // ── Stdlib trampoline direct-call declarations ──────────────
        // Signature conventions:
        //   0-arg: (env: PTR) -> (I64, I64)
        //   1-arg: (env: PTR, tag: I64, data: I64) -> (I64, I64)
        //   2-arg: (env: PTR, t1: I64, d1: I64, t2: I64, d2: I64) -> (I64, I64)
        //   3-arg: (env: PTR, t1-d1, t2-d2, t3-d3) -> (I64, I64)
        let sig0 = &[PTR];
        let sig1 = &[PTR, types::I64, types::I64];
        let sig2 = &[PTR, types::I64, types::I64, types::I64, types::I64];
        let sig3 = &[
            PTR,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ];
        let ret = &[types::I64, types::I64];

        // @"math" — 1-arg
        for name in &[
            "tok_math_sqrt_t",
            "tok_math_sin_t",
            "tok_math_cos_t",
            "tok_math_tan_t",
            "tok_math_asin_t",
            "tok_math_acos_t",
            "tok_math_atan_t",
            "tok_math_log_t",
            "tok_math_log2_t",
            "tok_math_log10_t",
            "tok_math_exp_t",
            "tok_math_floor_t",
            "tok_math_ceil_t",
            "tok_math_round_t",
            "tok_math_abs_t",
        ] {
            self.declare_runtime_func(name, sig1, ret);
        }
        // @"math" — 2-arg
        for name in &[
            "tok_math_pow_t",
            "tok_math_min_t",
            "tok_math_max_t",
            "tok_math_atan2_t",
        ] {
            self.declare_runtime_func(name, sig2, ret);
        }
        // @"math" — 0-arg
        self.declare_runtime_func("tok_math_random_t", sig0, ret);

        // @"str" — 1-arg
        for name in &[
            "tok_str_upper_t",
            "tok_str_lower_t",
            "tok_str_trim_t",
            "tok_str_trim_left_t",
            "tok_str_trim_right_t",
            "tok_str_chars_t",
            "tok_str_bytes_t",
            "tok_str_rev_t",
            "tok_str_len_t",
        ] {
            self.declare_runtime_func(name, sig1, ret);
        }
        // @"str" — 2-arg
        for name in &[
            "tok_str_contains_t",
            "tok_str_starts_with_t",
            "tok_str_ends_with_t",
            "tok_str_index_of_t",
            "tok_str_repeat_t",
            "tok_str_split_t",
        ] {
            self.declare_runtime_func(name, sig2, ret);
        }
        // @"str" — 3-arg
        for name in &[
            "tok_str_replace_t",
            "tok_str_pad_left_t",
            "tok_str_pad_right_t",
            "tok_str_substr_t",
        ] {
            self.declare_runtime_func(name, sig3, ret);
        }

        // @"json" — 1-arg
        for name in &[
            "tok_json_parse_t",
            "tok_json_stringify_t",
            "tok_json_pretty_t",
        ] {
            self.declare_runtime_func(name, sig1, ret);
        }

        // @"llm" — 1-arg
        self.declare_runtime_func("tok_llm_ask_t", sig1, ret);
        // @"llm" — 2-arg
        self.declare_runtime_func("tok_llm_chat_2_t", sig2, ret);

        // @"csv" — 1-arg
        for name in &["tok_csv_parse_t", "tok_csv_stringify_t"] {
            self.declare_runtime_func(name, sig1, ret);
        }

        // @"tmpl" — 1-arg
        self.declare_runtime_func("tok_tmpl_compile_t", sig1, ret);
        // @"tmpl" — 2-arg
        for name in &["tok_tmpl_render_t", "tok_tmpl_apply_t"] {
            self.declare_runtime_func(name, sig2, ret);
        }

        // @"toon" — 1-arg
        for name in &["tok_toon_parse_t", "tok_toon_stringify_t"] {
            self.declare_runtime_func(name, sig1, ret);
        }

        // @"os" — 0-arg
        for name in &["tok_os_args_t", "tok_os_cwd_t", "tok_os_pid_t"] {
            self.declare_runtime_func(name, sig0, ret);
        }
        // @"os" — 1-arg
        for name in &["tok_os_env_t", "tok_os_exit_t", "tok_os_exec_t"] {
            self.declare_runtime_func(name, sig1, ret);
        }
        // @"os" — 2-arg
        self.declare_runtime_func("tok_os_set_env_t", sig2, ret);

        // @"io" — 0-arg
        self.declare_runtime_func("tok_io_readall_t", sig0, ret);
        // @"io" — 1-arg (input with prompt; handles empty prompt for 0-arg case)
        self.declare_runtime_func("tok_io_input_1_t", sig1, ret);

        // @"fs" — 1-arg
        for name in &[
            "tok_fs_fread_t",
            "tok_fs_fexists_t",
            "tok_fs_fls_t",
            "tok_fs_fmk_t",
            "tok_fs_frm_t",
        ] {
            self.declare_runtime_func(name, sig1, ret);
        }
        // @"fs" — 2-arg
        for name in &["tok_fs_fwrite_t", "tok_fs_fappend_t"] {
            self.declare_runtime_func(name, sig2, ret);
        }

        // @"http" — 1-arg
        for name in &["tok_http_hget_t", "tok_http_hdel_t"] {
            self.declare_runtime_func(name, sig1, ret);
        }
        // @"http" — 2-arg
        for name in &["tok_http_hpost_t", "tok_http_hput_t", "tok_http_serve_t"] {
            self.declare_runtime_func(name, sig2, ret);
        }

        // @"re" — 2-arg
        for name in &["tok_re_rmatch_t", "tok_re_rfind_t", "tok_re_rall_t"] {
            self.declare_runtime_func(name, sig2, ret);
        }
        // @"re" — 3-arg
        self.declare_runtime_func("tok_re_rsub_t", sig3, ret);

        // @"time" — 0-arg
        self.declare_runtime_func("tok_time_now_t", sig0, ret);
        // @"time" — 1-arg
        self.declare_runtime_func("tok_time_sleep_t", sig1, ret);
        // @"time" — 2-arg
        self.declare_runtime_func("tok_time_fmt_t", sig2, ret);
    }

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

/// Compile a named function.
fn compile_function(
    compiler: &mut Compiler,
    name: &str,
    params: &[HirParam],
    ret_type: &Type,
    body: &[HirStmt],
) {
    let func_id = compiler.declare_tok_func(name, params, ret_type);
    let sig = compiler
        .module
        .declarations()
        .get_function_decl(func_id)
        .signature
        .clone();

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

    let mut func_ctx = FuncCtx {
        compiler,
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
        stdlib_imports: HashMap::new(),
        last_lambda_info: None,
        param_names: HashSet::new(),
        closure_sig_cache: HashMap::new(),
        tco_func_name: None,
        tco_loop_header: None,
        tco_param_vars: Vec::new(),
    };

    // Check for tail-call optimization opportunity
    let use_tco = is_self_tail_recursive(body, name);

    // Collect initial param values from entry block params
    let mut entry_param_vals = Vec::new();
    {
        let mut bpi = 0;
        for param in params.iter() {
            if matches!(param.ty, Type::Any) {
                let tag_val = func_ctx.builder.block_params(entry_block)[bpi];
                let data_val = func_ctx.builder.block_params(entry_block)[bpi + 1];
                entry_param_vals.push(tag_val);
                entry_param_vals.push(data_val);
                bpi += 2;
            } else if cl_type(&param.ty).is_some() {
                let val = func_ctx.builder.block_params(entry_block)[bpi];
                entry_param_vals.push(val);
                bpi += 1;
            }
        }
    }

    // If TCO, create loop header block and jump from entry
    let body_block = if use_tco {
        let loop_header = func_ctx.builder.create_block();
        // Add block params matching the function's Cranelift params
        for param in params.iter() {
            if matches!(param.ty, Type::Any) {
                func_ctx.builder.append_block_param(loop_header, types::I64); // tag
                func_ctx.builder.append_block_param(loop_header, types::I64); // data
            } else if let Some(ct) = cl_type(&param.ty) {
                func_ctx.builder.append_block_param(loop_header, ct);
            }
        }
        // Jump from entry to loop header with initial values
        func_ctx.builder.ins().jump(loop_header, &entry_param_vals);
        func_ctx.builder.switch_to_block(loop_header);
        // Don't seal loop_header yet — back-edges will be added by tail calls
        func_ctx.tco_func_name = Some(name.to_string());
        func_ctx.tco_loop_header = Some(loop_header);
        loop_header
    } else {
        entry_block
    };

    // Define parameters as variables
    // For Any params, each takes two block params (tag, data); we store as stack TokValue.
    let mut block_param_idx = 0;
    for param in params.iter() {
        func_ctx.param_names.insert(param.name.clone());
        if matches!(param.ty, Type::Any) {
            // Any param: two block params (tag, data), store as stack TokValue
            let tag_val = func_ctx.builder.block_params(body_block)[block_param_idx];
            let data_val = func_ctx.builder.block_params(body_block)[block_param_idx + 1];
            block_param_idx += 2;
            // Create stack slot and store
            let addr = alloc_tokvalue_on_stack(&mut func_ctx, tag_val, data_val);
            let var = func_ctx.new_var(PTR);
            func_ctx.builder.def_var(var, addr);
            func_ctx
                .vars
                .insert(param.name.clone(), (var, param.ty.clone()));
            func_ctx.tco_param_vars.push(var);
        } else if let Some(ct) = cl_type(&param.ty) {
            let var = func_ctx.new_var(ct);
            let param_val = func_ctx.builder.block_params(body_block)[block_param_idx];
            block_param_idx += 1;
            func_ctx.builder.def_var(var, param_val);
            func_ctx
                .vars
                .insert(param.name.clone(), (var, param.ty.clone()));
            func_ctx.tco_param_vars.push(var);
        }
    }

    // Compile body
    let last_val = compile_body(&mut func_ctx, body, ret_type);

    // Seal TCO loop header now that all back-edges have been added
    if let Some(loop_header) = func_ctx.tco_loop_header {
        func_ctx.builder.seal_block(loop_header);
    }

    // Jump to return block with value, but only if the current block isn't
    // already terminated (e.g., by a Return statement that already jumped).
    if !func_ctx.block_terminated {
        if is_any_return {
            if let Some(val) = last_val {
                // Determine the actual type of the last expression from the HIR,
                // so we use the correct type for to_tokvalue (not just ret_type=Any).
                let last_expr_ty = body
                    .last()
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

    // RC cleanup: dec all heap-typed locals (skip params — caller owns them)
    let heap_locals: Vec<(String, Variable, Type)> = func_ctx
        .vars
        .iter()
        .filter(|(name, (_, ty))| {
            !func_ctx.param_names.contains(name.as_str())
                && (is_heap_type(ty) || matches!(ty, Type::Any))
        })
        .map(|(name, (var, ty))| (name.clone(), *var, ty.clone()))
        .collect();

    if is_any_return {
        let tag_ret = func_ctx.builder.block_params(return_block)[0];
        let data_ret = func_ctx.builder.block_params(return_block)[1];

        // For Any return, data_ret might be a heap pointer.
        // Use aliasing guard: compare each local against data_ret, skip if same.
        for (_, var, ty) in &heap_locals {
            let v = func_ctx.builder.use_var(*var);
            // For Any-typed locals, the pointer is inside the TokValue stack slot;
            // for heap-typed locals, v is the pointer directly.
            let ptr = if matches!(ty, Type::Any) {
                // Load the data field (offset +8) from the stack TokValue
                func_ctx
                    .builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), v, 8)
            } else {
                v
            };
            let same = func_ctx.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal,
                ptr,
                data_ret,
            );
            let dec_block = func_ctx.builder.create_block();
            let cont_block = func_ctx.builder.create_block();
            func_ctx
                .builder
                .ins()
                .brif(same, cont_block, &[], dec_block, &[]);
            func_ctx.builder.switch_to_block(dec_block);
            func_ctx.builder.seal_block(dec_block);
            emit_rc_dec(&mut func_ctx, v, ty);
            func_ctx.builder.ins().jump(cont_block, &[]);
            func_ctx.builder.switch_to_block(cont_block);
            func_ctx.builder.seal_block(cont_block);
        }

        func_ctx.builder.ins().return_(&[tag_ret, data_ret]);
    } else if let Some(_rv) = return_var {
        let ret_val = func_ctx.builder.block_params(return_block)[0];

        if is_heap_type(ret_type) {
            // Return is a heap pointer — use aliasing guard per local
            for (_, var, ty) in &heap_locals {
                let v = func_ctx.builder.use_var(*var);
                let same = func_ctx.builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::Equal,
                    v,
                    ret_val,
                );
                let dec_block = func_ctx.builder.create_block();
                let cont_block = func_ctx.builder.create_block();
                func_ctx
                    .builder
                    .ins()
                    .brif(same, cont_block, &[], dec_block, &[]);
                func_ctx.builder.switch_to_block(dec_block);
                func_ctx.builder.seal_block(dec_block);
                emit_rc_dec(&mut func_ctx, v, ty);
                func_ctx.builder.ins().jump(cont_block, &[]);
                func_ctx.builder.switch_to_block(cont_block);
                func_ctx.builder.seal_block(cont_block);
            }
        } else {
            // Return is scalar — safe to unconditionally dec all heap locals
            for (_, var, ty) in &heap_locals {
                let v = func_ctx.builder.use_var(*var);
                emit_rc_dec(&mut func_ctx, v, ty);
            }
        }

        func_ctx.builder.ins().return_(&[ret_val]);
    } else {
        // Void return: unconditionally dec all heap locals
        for (_, var, ty) in &heap_locals {
            let v = func_ctx.builder.use_var(*var);
            emit_rc_dec(&mut func_ctx, v, ty);
        }
        func_ctx.builder.ins().return_(&[]);
    }

    func_ctx.builder.finalize();

    let mut ctx = Context::for_function(func);
    compiler
        .module
        .define_function(func_id, &mut ctx)
        .expect("codegen: failed to define function");
}

/// Compile a deferred lambda body into its own Cranelift function.
///
/// Lambda calling convention: (env_ptr: PTR, arg0_tag: I64, arg0_data: I64, ...) -> (I64, I64)
/// All params are passed as TokValue (tag, data) pairs, return is a TokValue pair.
fn compile_lambda_body(compiler: &mut Compiler, lambda: &PendingLambda) {
    let sig = compiler
        .module
        .declarations()
        .get_function_decl(lambda.func_id)
        .signature
        .clone();

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

    let mut func_ctx = FuncCtx {
        compiler,
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
        stdlib_imports: HashMap::new(),
        last_lambda_info: None,
        param_names: HashSet::new(),
        closure_sig_cache: HashMap::new(),
        tco_func_name: None,
        tco_loop_header: None,
        tco_param_vars: Vec::new(),
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
        let tag = func_ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), env_ptr_val, offset);
        let data =
            func_ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), env_ptr_val, offset + 8);
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
            let last_expr_ty = lambda
                .body
                .last()
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
        .expect("codegen: failed to define lambda function");
}

/// Compile a specialized lambda body with native-typed calling convention.
///
/// Specialized calling convention: (env_ptr: PTR, arg0: T0, arg1: T1, ...) -> RetT
/// Params are native types, no boxing/unboxing.
fn compile_specialized_lambda_body(compiler: &mut Compiler, lambda: &PendingLambda) {
    let spec_types = lambda
        .specialized_param_types
        .as_ref()
        .expect("codegen: specialized lambda missing param types");
    let sig = compiler
        .module
        .declarations()
        .get_function_decl(lambda.func_id)
        .signature
        .clone();

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

    let mut func_ctx = FuncCtx {
        compiler,
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
        stdlib_imports: HashMap::new(),
        last_lambda_info: None,
        param_names: HashSet::new(),
        closure_sig_cache: HashMap::new(),
        tco_func_name: None,
        tco_loop_header: None,
        tco_param_vars: Vec::new(),
    };

    // Bind parameters: first block param is env_ptr, then one native value per param
    let env_ptr_val = func_ctx.builder.block_params(entry_block)[0];
    for (i, (param, param_ty)) in lambda.params.iter().zip(spec_types.iter()).enumerate() {
        let val = func_ctx.builder.block_params(entry_block)[1 + i];
        let ct = cl_type_or_i64(param_ty);
        let var = func_ctx.new_var(ct);
        func_ctx.builder.def_var(var, val);
        func_ctx
            .vars
            .insert(param.name.clone(), (var, param_ty.clone()));
    }

    // Load captured variables from env_ptr
    for (i, cap) in lambda.captures.iter().enumerate() {
        let offset = (i * 16) as i32;
        match &cap.ty {
            Type::Int => {
                // Extract data field directly as i64 (skip tag)
                let data = func_ctx.builder.ins().load(
                    types::I64,
                    MemFlags::trusted(),
                    env_ptr_val,
                    offset + 8,
                );
                let var = func_ctx.new_var(types::I64);
                func_ctx.builder.def_var(var, data);
                func_ctx.vars.insert(cap.name.clone(), (var, Type::Int));
            }
            Type::Float => {
                let data = func_ctx.builder.ins().load(
                    types::I64,
                    MemFlags::trusted(),
                    env_ptr_val,
                    offset + 8,
                );
                let fval = func_ctx
                    .builder
                    .ins()
                    .bitcast(types::F64, MemFlags::new(), data);
                let var = func_ctx.new_var(types::F64);
                func_ctx.builder.def_var(var, fval);
                func_ctx.vars.insert(cap.name.clone(), (var, Type::Float));
            }
            Type::Bool => {
                let data = func_ctx.builder.ins().load(
                    types::I64,
                    MemFlags::trusted(),
                    env_ptr_val,
                    offset + 8,
                );
                let bval = func_ctx.builder.ins().ireduce(types::I8, data);
                let var = func_ctx.new_var(types::I8);
                func_ctx.builder.def_var(var, bval);
                func_ctx.vars.insert(cap.name.clone(), (var, Type::Bool));
            }
            _ => {
                // Unknown type — load as Any (TokValue on stack)
                let tag = func_ctx.builder.ins().load(
                    types::I64,
                    MemFlags::trusted(),
                    env_ptr_val,
                    offset,
                );
                let data = func_ctx.builder.ins().load(
                    types::I64,
                    MemFlags::trusted(),
                    env_ptr_val,
                    offset + 8,
                );
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
            let last_expr_ty = lambda
                .body
                .last()
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
        .expect("codegen: failed to define specialized lambda");
}

/// Compile top-level statements into `_tok_main`.
fn compile_main(compiler: &mut Compiler, stmts: &[HirStmt]) {
    let sig = compiler.module.make_signature();
    // _tok_main returns void
    let func_id = compiler
        .module
        .declare_function("_tok_main", Linkage::Export, &sig)
        .expect("codegen: failed to declare _tok_main");

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, func_id.as_u32());

    let mut func_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_builder_ctx);

    let entry_block = builder.create_block();
    let return_block = builder.create_block();

    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let mut func_ctx = FuncCtx {
        compiler,
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
        stdlib_imports: HashMap::new(),
        last_lambda_info: None,
        param_names: HashSet::new(),
        closure_sig_cache: HashMap::new(),
        tco_func_name: None,
        tco_loop_header: None,
        tco_param_vars: Vec::new(),
    };

    compile_body(&mut func_ctx, stmts, &Type::Nil);

    if !func_ctx.block_terminated {
        func_ctx.builder.ins().jump(return_block, &[]);
    }
    func_ctx.builder.switch_to_block(return_block);
    func_ctx.builder.seal_block(return_block);

    // RC cleanup: dec all heap-typed locals before exit (main has no params)
    let heap_vars: Vec<(Variable, Type)> = func_ctx
        .vars
        .values()
        .filter(|(_, ty)| is_heap_type(ty) || matches!(ty, Type::Any))
        .map(|(var, ty)| (*var, ty.clone()))
        .collect();
    for (var, ty) in &heap_vars {
        let v = func_ctx.builder.use_var(*var);
        emit_rc_dec(&mut func_ctx, v, ty);
    }

    func_ctx.builder.ins().return_(&[]);
    func_ctx.builder.finalize();

    let mut ctx = Context::for_function(func);
    if std::env::var("CLIF_DUMP").is_ok() {
        eprintln!("=== _tok_main IR ===\n{}", ctx.func.display());
    }
    compiler
        .module
        .define_function(func_id, &mut ctx)
        .expect("codegen: failed to define function");
}

/// Compile the C `main` entry point that calls `_tok_main`.
fn compile_entry(compiler: &mut Compiler) {
    let mut sig = compiler.module.make_signature();
    sig.params.push(AbiParam::new(types::I32)); // argc
    sig.params.push(AbiParam::new(PTR)); // argv
    sig.returns.push(AbiParam::new(types::I32)); // exit code

    let func_id = compiler
        .module
        .declare_function("main", Linkage::Export, &sig)
        .expect("codegen: failed to declare main");

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, func_id.as_u32());

    // Declare _tok_main reference before creating builder (avoids borrow conflict)
    let tok_main_sig = compiler.module.make_signature();
    let tok_main_id = compiler
        .module
        .declare_function("_tok_main", Linkage::Export, &tok_main_sig)
        .expect("codegen: failed to declare _tok_main reference");
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
        .expect("codegen: failed to define function");
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
fn compile_body(ctx: &mut FuncCtx, stmts: &[HirStmt], _expected_type: &Type) -> Option<Value> {
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
                ctx.known_closures.insert(
                    name.clone(),
                    KnownClosure {
                        func_id,
                        env_ptr,
                        pending_idx,
                        specialized: None,
                    },
                );
            } else {
                // Variable reassigned to non-lambda — invalidate
                ctx.known_closures.remove(name.as_str());
            }
            // Track stdlib module imports for direct-call optimization
            if let HirExprKind::RuntimeCall {
                name: ref call_name,
                ref args,
            } = value.kind
            {
                if call_name == "tok_import" {
                    if let Some(arg) = args.first() {
                        if let HirExprKind::Str(module_name) = &arg.kind {
                            ctx.stdlib_imports.insert(name.clone(), module_name.clone());
                        }
                    }
                }
            } else {
                ctx.stdlib_imports.remove(name.as_str());
            }
            if let Some(v) = val {
                if let Some((var, existing_ty)) = ctx.vars.get(name).cloned() {
                    // Save old value for RC dec after computing new value
                    let needs_rc = is_heap_type(&existing_ty) || matches!(existing_ty, Type::Any);
                    let old_val = if needs_rc {
                        Some(ctx.builder.use_var(var))
                    } else {
                        None
                    };
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
                    // Determine the effective type after coercion.
                    // Coercion arms 1-5 produce values matching existing_ty.
                    // The catch-all (_ => v) passes through the raw value, which
                    // may have a different type — track it so cleanup uses the
                    // correct type for RC dec.
                    let effective_ty = match (&existing_ty, &value.ty) {
                        // Arms that extract concrete from Any → stays existing_ty
                        (Type::Int, Type::Any)
                        | (Type::Float, Type::Any)
                        | (Type::Bool, Type::Any) => existing_ty.clone(),
                        (_, Type::Any) if !matches!(existing_ty, Type::Any) => existing_ty.clone(),
                        // Wrap concrete→Any → stays Any
                        (Type::Any, vt) if !matches!(vt, Type::Any | Type::Nil | Type::Never) => {
                            existing_ty.clone()
                        }
                        // Catch-all: actual type may have changed
                        _ => value.ty.clone(),
                    };

                    let new_ct = cl_type_or_i64(&effective_ty);
                    let old_ct = cl_type_or_i64(&existing_ty);
                    let type_changed = std::mem::discriminant(&effective_ty)
                        != std::mem::discriminant(&existing_ty);

                    if new_ct != old_ct {
                        // Cranelift type changed (e.g., F64↔I64) — need a new variable
                        let new_var = ctx.new_var(new_ct);
                        ctx.builder.def_var(new_var, coerced);
                        ctx.vars.insert(name.clone(), (new_var, effective_ty));
                    } else {
                        ctx.builder.def_var(var, coerced);
                        if type_changed {
                            ctx.vars.insert(name.clone(), (var, effective_ty));
                        }
                    }

                    // RC: decrement old value now that new value is stored.
                    if let Some(old) = old_val {
                        if type_changed {
                            // Types differ, so values can't alias — always dec old
                            emit_rc_dec(ctx, old, &existing_ty);
                        } else if is_heap_type(&existing_ty) {
                            // Same concrete heap type: only dec if old != new pointer
                            // (guards against aliasing, e.g., x = push(x, v))
                            let same = ctx.builder.ins().icmp(
                                cranelift_codegen::ir::condcodes::IntCC::Equal,
                                old,
                                coerced,
                            );
                            let dec_block = ctx.builder.create_block();
                            let cont_block = ctx.builder.create_block();
                            ctx.builder
                                .ins()
                                .brif(same, cont_block, &[], dec_block, &[]);

                            ctx.builder.switch_to_block(dec_block);
                            ctx.builder.seal_block(dec_block);
                            emit_rc_dec(ctx, old, &existing_ty);
                            ctx.builder.ins().jump(cont_block, &[]);

                            ctx.builder.switch_to_block(cont_block);
                            ctx.builder.seal_block(cont_block);
                        } else {
                            // Any type: always dec (Any values are stack TokValues,
                            // not raw pointers, so aliasing is rare and the runtime
                            // handles tag-based no-op for scalars)
                            emit_rc_dec(ctx, old, &existing_ty);
                        }
                    }
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
            let target_val =
                compile_expr(ctx, target).expect("codegen: target expr produced no value");
            let idx_val = compile_expr(ctx, index).expect("codegen: index expr produced no value");
            let val = compile_expr(ctx, value).expect("codegen: value expr produced no value");
            // Call appropriate runtime set function based on target type
            match &target.ty {
                Type::Array(_) => {
                    // Pack value as TokValue for array_set
                    let idx = unwrap_any_ptr(ctx, idx_val, &index.ty);
                    let (tag, data) = to_tokvalue(ctx, val, &value.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_set");
                    ctx.builder
                        .ins()
                        .call(func_ref, &[target_val, idx, tag, data]);
                }
                Type::Map(_) => {
                    // Key must be a string pointer; unwrap from Any if needed
                    let key = unwrap_any_ptr(ctx, idx_val, &index.ty);
                    let (tag, data) = to_tokvalue(ctx, val, &value.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_map_set");
                    ctx.builder
                        .ins()
                        .call(func_ref, &[target_val, key, tag, data]);
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
            let target_val =
                compile_expr(ctx, target).expect("codegen: target expr produced no value");
            let val = compile_expr(ctx, value).expect("codegen: value expr produced no value");
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
            ctx.builder
                .ins()
                .call(set_ref, &[target_val, key_str, tag, data]);
            None
        }

        HirStmt::Expr(expr) => compile_expr(ctx, expr),

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
                } else if ctx.is_any_return {
                    let zero = ctx.builder.ins().iconst(types::I64, 0);
                    ctx.builder.ins().jump(ctx.return_block, &[zero, zero]);
                } else {
                    ctx.builder.ins().jump(ctx.return_block, &[]);
                }
            } else if ctx.is_any_return {
                let zero = ctx.builder.ins().iconst(types::I64, 0);
                ctx.builder.ins().jump(ctx.return_block, &[zero, zero]);
            } else {
                ctx.builder.ins().jump(ctx.return_block, &[]);
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
                let dead_block = ctx.builder.create_block();
                ctx.builder.switch_to_block(dead_block);
                ctx.builder.seal_block(dead_block);
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
        HirExprKind::Int(n) => Some(ctx.builder.ins().iconst(types::I64, *n)),

        HirExprKind::Float(f) => Some(ctx.builder.ins().f64const(*f)),

        HirExprKind::Bool(b) => Some(ctx.builder.ins().iconst(types::I8, *b as i64)),

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
                if matches!(var_ty, Type::Any)
                    && !matches!(&expr.ty, Type::Any | Type::Nil | Type::Never)
                {
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
                let (param_types, ret_type) = ctx
                    .compiler
                    .func_sigs
                    .get(name.as_str())
                    .expect("codegen: unknown function signature")
                    .clone();
                let trampoline_name = format!("__tok_tramp_{}", name);

                // Check if trampoline already exists (avoid duplicate definition)
                let tramp_func_id = if let Some(&existing) =
                    ctx.compiler.declared_funcs.get(trampoline_name.as_str())
                {
                    existing
                } else {
                    // Create trampoline as a PendingLambda that just calls the function
                    let tramp_params: Vec<HirParam> = param_types
                        .iter()
                        .enumerate()
                        .map(|(i, ty)| HirParam {
                            name: format!("__p{}", i),
                            ty: ty.clone(),
                            variadic: false,
                        })
                        .collect();
                    let call_args: Vec<HirExpr> = tramp_params
                        .iter()
                        .map(|p| HirExpr::new(HirExprKind::Ident(p.name.clone()), p.ty.clone()))
                        .collect();
                    let call_expr = HirExpr::new(
                        HirExprKind::Call {
                            func: Box::new(HirExpr::new(
                                HirExprKind::Ident(name.clone()),
                                Type::Any,
                            )),
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

                    let fid = ctx
                        .compiler
                        .module
                        .declare_function(&trampoline_name, Linkage::Local, &sig)
                        .expect("codegen: failed to declare trampoline");

                    ctx.compiler
                        .declared_funcs
                        .insert(trampoline_name.clone(), fid);

                    ctx.compiler.pending_lambdas.push(PendingLambda {
                        name: trampoline_name.clone(),
                        func_id: fid,
                        params: tramp_params,
                        ret_type: ret_type.clone(),
                        body: vec![HirStmt::Expr(call_expr)],
                        captures: vec![],
                        specialized_param_types: None,
                    });

                    fid
                };

                let tramp_ref = ctx
                    .compiler
                    .module
                    .declare_func_in_func(tramp_func_id, ctx.builder.func);
                let fn_ptr = ctx.builder.ins().func_addr(PTR, tramp_ref);
                let env_ptr = ctx.builder.ins().iconst(PTR, 0);
                let arity_val = ctx
                    .builder
                    .ins()
                    .iconst(types::I32, param_types.len() as i64);
                let alloc_ref = ctx.get_runtime_func_ref("tok_closure_alloc");
                let call = ctx
                    .builder
                    .ins()
                    .call(alloc_ref, &[fn_ptr, env_ptr, arity_val]);
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
                let val = compile_expr(ctx, elem)
                    .unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
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

                let val = compile_expr(ctx, val_expr)
                    .unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
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
                let val = compile_expr(ctx, elem)
                    .unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
                let idx = ctx.builder.ins().iconst(types::I64, i as i64);
                let (tag, data) = to_tokvalue(ctx, val, &elem.ty);
                let set_ref = ctx.get_runtime_func_ref("tok_tuple_set");
                ctx.builder.ins().call(set_ref, &[tuple, idx, tag, data]);
            }
            Some(tuple)
        }

        HirExprKind::BinOp { op, left, right } => compile_binop(ctx, *op, left, right, &expr.ty),

        HirExprKind::UnaryOp { op, operand } => compile_unaryop(ctx, *op, operand, &expr.ty),

        HirExprKind::Index { target, index } => {
            let target_val =
                compile_expr(ctx, target).expect("codegen: target expr produced no value");
            let idx_val = compile_expr(ctx, index).expect("codegen: index expr produced no value");
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
            // ── Stdlib constant inlining (e.g. m.pi, m.e, m.inf, m.nan) ──
            if let HirExprKind::Ident(var_name) = &target.kind {
                if let Some(module_name) = ctx.stdlib_imports.get(var_name.as_str()).cloned() {
                    if let Some(const_val) = get_stdlib_const(&module_name, field) {
                        let f_val = ctx.builder.ins().f64const(const_val);
                        let (tag, data) = to_tokvalue(ctx, f_val, &Type::Float);
                        return Some(from_tokvalue(ctx, tag, data, &expr.ty));
                    }
                }
            }
            let target_val =
                compile_expr(ctx, target).expect("codegen: target expr produced no value");
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
                    let map_ptr =
                        ctx.builder
                            .ins()
                            .load(types::I64, MemFlags::trusted(), target_val, 8);
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

        HirExprKind::Call { func, args } => compile_call(ctx, func, args, &expr.ty),

        HirExprKind::RuntimeCall { name, args } => {
            // Special-case filter/reduce: closure arg needs special handling
            match name.as_str() {
                "tok_import" => {
                    // Stdlib module import: intercept known module names
                    if let HirExprKind::Str(path) = &args[0].kind {
                        let constructor = match path.as_str() {
                            "math" => "tok_stdlib_math",
                            "str"  => "tok_stdlib_str",
                            "os"   => "tok_stdlib_os",
                            "io"   => "tok_stdlib_io",
                            "json" => "tok_stdlib_json",
                            "llm"  => "tok_stdlib_llm",
                            "csv"  => "tok_stdlib_csv",
                            "fs"   => "tok_stdlib_fs",
                            "http" => "tok_stdlib_http",
                            "re"   => "tok_stdlib_re",
                            "time" => "tok_stdlib_time",
                            "tmpl" => "tok_stdlib_tmpl",
                            "toon" => "tok_stdlib_toon",
                            other  => panic!("Unknown module: @\"{}\" — only stdlib modules (math, str, os, io, json, csv, fs, http, re, time, tmpl, toon, llm) are supported in compiled mode", other),
                        };
                        let func_ref = ctx.get_runtime_func_ref(constructor);
                        let call = ctx.builder.ins().call(func_ref, &[]);
                        return Some(ctx.builder.inst_results(call)[0]);
                    }
                    panic!("Dynamic imports not supported in compiled mode");
                }
                "tok_array_filter" => {
                    // Inline filter when lambda is literal and element type is concrete
                    if can_inline_hof(&args[1], &args[0].ty, 1) {
                        return compile_inline_filter(ctx, &args[0], &args[1], &expr.ty);
                    }
                    // Fallback: runtime call
                    let arr_raw =
                        compile_expr(ctx, &args[0]).expect("codegen: args[0] produced no value");
                    let arr =
                        if matches!(&args[0].ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                            ctx.builder
                                .ins()
                                .load(types::I64, MemFlags::trusted(), arr_raw, 8)
                        } else {
                            arr_raw
                        };
                    let closure =
                        compile_expr(ctx, &args[1]).expect("codegen: args[1] produced no value");
                    let func_ref = ctx.get_runtime_func_ref("tok_array_filter");
                    let call = ctx.builder.ins().call(func_ref, &[arr, closure]);
                    let result = ctx.builder.inst_results(call)[0];
                    if matches!(&expr.ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                        let tag = ctx.builder.ins().iconst(types::I64, TAG_ARRAY);
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
                    let arr_raw =
                        compile_expr(ctx, &args[0]).expect("codegen: args[0] produced no value");
                    let arr =
                        if matches!(&args[0].ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                            ctx.builder
                                .ins()
                                .load(types::I64, MemFlags::trusted(), arr_raw, 8)
                        } else {
                            arr_raw
                        };
                    let init_val = compile_expr(ctx, &args[1]);
                    let closure =
                        compile_expr(ctx, &args[2]).expect("codegen: args[2] produced no value");
                    let (init_tag, init_data) = if let Some(iv) = init_val {
                        to_tokvalue(ctx, iv, &args[1].ty)
                    } else {
                        let zero = ctx.builder.ins().iconst(types::I64, 0);
                        (zero, zero)
                    };
                    let func_ref = ctx.get_runtime_func_ref("tok_array_reduce");
                    let call = ctx
                        .builder
                        .ins()
                        .call(func_ref, &[arr, init_tag, init_data, closure]);
                    let results = ctx.builder.inst_results(call);
                    return Some(from_tokvalue(ctx, results[0], results[1], &expr.ty));
                }
                "tok_array_push" => {
                    // tok_array_push(arr: PTR, tag: I64, data: I64) -> PTR
                    let arr =
                        compile_expr(ctx, &args[0]).expect("codegen: args[0] produced no value");
                    let val =
                        compile_expr(ctx, &args[1]).expect("codegen: args[1] produced no value");
                    let (tag, data) = to_tokvalue(ctx, val, &args[1].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_push");
                    let call = ctx.builder.ins().call(func_ref, &[arr, tag, data]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
                "tok_value_to_string" => {
                    // tok_value_to_string(tag: I64, data: I64) -> PTR
                    // HIR emits 1 arg but runtime expects (tag, data)
                    let val =
                        compile_expr(ctx, &args[0]).expect("codegen: args[0] produced no value");
                    let (tag, data) = to_tokvalue(ctx, val, &args[0].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_value_to_string");
                    let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
                "tok_array_concat" => {
                    // tok_array_concat(a: PTR, b: PTR) -> PTR
                    // Both args need to be unwrapped from Any if needed
                    let a_raw =
                        compile_expr(ctx, &args[0]).expect("codegen: args[0] produced no value");
                    let a =
                        if matches!(&args[0].ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                            ctx.builder
                                .ins()
                                .load(types::I64, MemFlags::trusted(), a_raw, 8)
                        } else {
                            a_raw
                        };
                    let b_raw =
                        compile_expr(ctx, &args[1]).expect("codegen: args[1] produced no value");
                    let b =
                        if matches!(&args[1].ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                            ctx.builder
                                .ins()
                                .load(types::I64, MemFlags::trusted(), b_raw, 8)
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

        HirExprKind::Lambda {
            params,
            ret_type,
            body,
        } => {
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

            let func_id = ctx
                .compiler
                .module
                .declare_function(&lambda_name, Linkage::Local, &sig)
                .expect("codegen: failed to declare lambda");

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
            let func_ref = ctx
                .compiler
                .module
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
                    let (var, var_ty) = ctx
                        .vars
                        .get(&cap.name)
                        .expect("codegen: captured var not found")
                        .clone();
                    let val = ctx.builder.use_var(var);
                    let (tag, data) = to_tokvalue(ctx, val, &var_ty);
                    let offset = (i * 16) as i32;
                    ctx.builder
                        .ins()
                        .store(MemFlags::trusted(), tag, env, offset);
                    ctx.builder
                        .ins()
                        .store(MemFlags::trusted(), data, env, offset + 8);
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
        } => compile_if(
            ctx, cond, then_body, then_expr, else_body, else_expr, &expr.ty,
        ),

        HirExprKind::Loop { kind, body } => {
            compile_loop(ctx, kind, body);
            None
        }

        HirExprKind::Block {
            stmts,
            expr: block_expr,
        } => {
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
            let target_val =
                compile_expr(ctx, target).expect("codegen: target expr produced no value");
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

        HirExprKind::Range {
            start,
            end: _,
            inclusive: _,
        } => {
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
            collect_free_vars_expr(body_expr, &empty_locals, &mut free_set, 0);
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

            let func_id = ctx
                .compiler
                .module
                .declare_function(&thunk_name, Linkage::Local, &sig)
                .expect("codegen: failed to declare go thunk");

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
            let func_ref = ctx
                .compiler
                .module
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
                    let (var, var_ty) = ctx
                        .vars
                        .get(&cap.name)
                        .expect("codegen: captured var not found")
                        .clone();
                    let val = ctx.builder.use_var(var);
                    let (tag, data) = to_tokvalue(ctx, val, &var_ty);
                    let offset = (i * 16) as i32;
                    ctx.builder
                        .ins()
                        .store(MemFlags::trusted(), tag, env, offset);
                    ctx.builder
                        .ins()
                        .store(MemFlags::trusted(), data, env, offset + 8);
                }
                env
            };

            // Call tok_go(fn_ptr, env_ptr) -> *mut TokHandle
            let go_ref = ctx.get_runtime_func_ref("tok_go");
            let call = ctx.builder.ins().call(go_ref, &[fn_ptr, env_ptr]);
            Some(ctx.builder.inst_results(call)[0])
        }

        HirExprKind::Receive(chan_expr) => {
            let chan =
                compile_expr(ctx, chan_expr).expect("codegen: channel expr produced no value");
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
            let chan_val =
                compile_expr(ctx, chan).expect("codegen: channel expr produced no value");
            let val = compile_expr(ctx, value).expect("codegen: value expr produced no value");
            let (tag, data) = to_tokvalue(ctx, val, &value.ty);
            let func_ref = ctx.get_runtime_func_ref("tok_channel_send");
            ctx.builder.ins().call(func_ref, &[chan_val, tag, data]);
            None
        }

        HirExprKind::Select(arms) => {
            // Select: try each arm non-blocking in order.
            // If an arm succeeds, execute its body and jump to merge.
            // If no arm succeeds:
            //   - If there's a default arm, execute it
            //   - Otherwise, block on the first recv arm
            let merge_block = ctx.builder.create_block();

            // Separate default arm from channel arms
            let mut default_body: Option<&Vec<HirStmt>> = None;
            let mut channel_arms: Vec<&HirSelectArm> = Vec::new();
            for arm in arms.iter() {
                match arm {
                    HirSelectArm::Default(body) => default_body = Some(body),
                    _ => channel_arms.push(arm),
                }
            }

            // Try each channel arm non-blocking
            for arm in channel_arms.iter() {
                let next_block = ctx.builder.create_block();
                let body_block = ctx.builder.create_block();

                match arm {
                    HirSelectArm::Recv { var, chan, body } => {
                        let chan_val = compile_expr(ctx, chan)
                            .expect("codegen: channel expr produced no value");
                        // Allocate stack slot for try_recv output
                        let ss = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                            StackSlotKind::ExplicitSlot,
                            16, // sizeof(TokValue)
                            3,  // 8-byte alignment
                        ));
                        let out_ptr = ctx.builder.ins().stack_addr(PTR, ss, 0);
                        let try_recv_ref = ctx.get_runtime_func_ref("tok_channel_try_recv");
                        let call = ctx.builder.ins().call(try_recv_ref, &[chan_val, out_ptr]);
                        let ok = ctx.builder.inst_results(call)[0];
                        ctx.builder.ins().brif(ok, body_block, &[], next_block, &[]);

                        // Body block: received successfully
                        // out_ptr is a pointer to TokValue on the stack — use it as Any
                        ctx.builder.switch_to_block(body_block);
                        ctx.builder.seal_block(body_block);
                        let ct = cl_type_or_i64(&Type::Any);
                        let v = ctx.new_var(ct);
                        ctx.builder.def_var(v, out_ptr);
                        ctx.vars.insert(var.clone(), (v, Type::Any));
                        ctx.block_terminated = false;
                        compile_body(ctx, body, &Type::Nil);
                        if !ctx.block_terminated {
                            ctx.builder.ins().jump(merge_block, &[]);
                        }
                    }
                    HirSelectArm::Send { chan, value, body } => {
                        let chan_val = compile_expr(ctx, chan)
                            .expect("codegen: channel expr produced no value");
                        let val = compile_expr(ctx, value)
                            .expect("codegen: value expr produced no value");
                        let (tag, data) = to_tokvalue(ctx, val, &value.ty);
                        let try_send_ref = ctx.get_runtime_func_ref("tok_channel_try_send");
                        let call = ctx.builder.ins().call(try_send_ref, &[chan_val, tag, data]);
                        let ok = ctx.builder.inst_results(call)[0];
                        ctx.builder.ins().brif(ok, body_block, &[], next_block, &[]);

                        // Body block: sent successfully
                        ctx.builder.switch_to_block(body_block);
                        ctx.builder.seal_block(body_block);
                        ctx.block_terminated = false;
                        compile_body(ctx, body, &Type::Nil);
                        if !ctx.block_terminated {
                            ctx.builder.ins().jump(merge_block, &[]);
                        }
                    }
                    HirSelectArm::Default(_) => unreachable!(),
                }

                ctx.builder.switch_to_block(next_block);
                ctx.builder.seal_block(next_block);
            }

            // All non-blocking attempts failed — we're in the fallthrough block
            if let Some(body) = default_body {
                // Default arm exists: execute it
                ctx.block_terminated = false;
                compile_body(ctx, body, &Type::Nil);
                if !ctx.block_terminated {
                    ctx.builder.ins().jump(merge_block, &[]);
                }
            } else if let Some(first_recv) = channel_arms
                .iter()
                .find(|a| matches!(a, HirSelectArm::Recv { .. }))
            {
                // No default: block on first recv arm
                if let HirSelectArm::Recv { var, chan, body } = first_recv {
                    let chan_val =
                        compile_expr(ctx, chan).expect("codegen: channel expr produced no value");
                    let recv_ref = ctx.get_runtime_func_ref("tok_channel_recv");
                    let call = ctx.builder.ins().call(recv_ref, &[chan_val]);
                    let results = ctx.builder.inst_results(call);
                    let tag = results[0];
                    let data = results[1];
                    // Store as TokValue on stack for Any-typed access
                    let val_ptr = alloc_tokvalue_on_stack(ctx, tag, data);
                    let ct = cl_type_or_i64(&Type::Any);
                    let v = ctx.new_var(ct);
                    ctx.builder.def_var(v, val_ptr);
                    ctx.vars.insert(var.clone(), (v, Type::Any));
                    ctx.block_terminated = false;
                    compile_body(ctx, body, &Type::Nil);
                    if !ctx.block_terminated {
                        ctx.builder.ins().jump(merge_block, &[]);
                    }
                }
            } else {
                // No recv arms and no default — just jump to merge
                ctx.builder.ins().jump(merge_block, &[]);
            }

            ctx.builder.switch_to_block(merge_block);
            ctx.builder.seal_block(merge_block);
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

    let left_is_any = matches!(&left.ty, Type::Any | Type::Optional(_) | Type::Result(_));
    let right_is_any = matches!(&right.ty, Type::Any | Type::Optional(_) | Type::Result(_));

    // If both sides are Int (and neither is Any)
    if matches!(left.ty, Type::Int)
        && matches!(right.ty, Type::Int)
        && !left_is_any
        && !right_is_any
    {
        return compile_int_binop(ctx, op, lv, rv);
    }

    // If both sides are Float (or one is Float and one is Int), but neither is Any
    if !left_is_any
        && !right_is_any
        && (matches!(left.ty, Type::Float) || matches!(right.ty, Type::Float))
    {
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
    if matches!(left.ty, Type::Str) && matches!(right.ty, Type::Str) && matches!(op, HirBinOp::Add)
    {
        let func_ref = ctx.get_runtime_func_ref("tok_string_concat");
        let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
        return Some(ctx.builder.inst_results(call)[0]);
    }

    // String multiplication: "ha" * 3 or 3 * "ha"
    if matches!(op, HirBinOp::Mul) {
        if matches!(left.ty, Type::Str) && matches!(right.ty, Type::Int) {
            let func_ref = ctx.get_runtime_func_ref("tok_string_repeat");
            let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
        if matches!(left.ty, Type::Int) && matches!(right.ty, Type::Str) {
            let func_ref = ctx.get_runtime_func_ref("tok_string_repeat");
            let call = ctx.builder.ins().call(func_ref, &[rv, lv]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
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
                    HirBinOp::LtEq => {
                        cranelift_codegen::ir::condcodes::IntCC::SignedLessThanOrEqual
                    }
                    HirBinOp::GtEq => {
                        cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThanOrEqual
                    }
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
                let result =
                    ctx.builder
                        .ins()
                        .icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, lv, rv);
                return Some(result);
            }
            HirBinOp::Neq => {
                let result = ctx.builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                    lv,
                    rv,
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
        HirBinOp::Pow => "tok_value_pow",
        _ => {
            // Bitwise ops — fallback to 0
            return Some(ctx.builder.ins().iconst(types::I64, 0));
        }
    };
    let func_ref = ctx.get_runtime_func_ref(rt_name);
    let call = ctx.builder.ins().call(func_ref, &[lt, ld, rt, rd]);
    let results = ctx.builder.inst_results(call);
    Some(from_tokvalue(ctx, results[0], results[1], result_ty))
}

fn compile_int_binop(ctx: &mut FuncCtx, op: HirBinOp, lv: Value, rv: Value) -> Option<Value> {
    use cranelift_codegen::ir::condcodes::IntCC;
    Some(match op {
        HirBinOp::Add => ctx.builder.ins().iadd(lv, rv),
        HirBinOp::Sub => ctx.builder.ins().isub(lv, rv),
        HirBinOp::Mul => ctx.builder.ins().imul(lv, rv),
        HirBinOp::Div => ctx.builder.ins().sdiv(lv, rv),
        HirBinOp::Mod => ctx.builder.ins().srem(lv, rv),
        HirBinOp::Pow => {
            let func_ref = ctx.get_runtime_func_ref("tok_pow_int");
            let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
            ctx.builder.inst_results(call)[0]
        }
        HirBinOp::Eq => ctx.builder.ins().icmp(IntCC::Equal, lv, rv),
        HirBinOp::Neq => ctx.builder.ins().icmp(IntCC::NotEqual, lv, rv),
        HirBinOp::Lt => ctx.builder.ins().icmp(IntCC::SignedLessThan, lv, rv),
        HirBinOp::Gt => ctx.builder.ins().icmp(IntCC::SignedGreaterThan, lv, rv),
        HirBinOp::LtEq => ctx.builder.ins().icmp(IntCC::SignedLessThanOrEqual, lv, rv),
        HirBinOp::GtEq => ctx
            .builder
            .ins()
            .icmp(IntCC::SignedGreaterThanOrEqual, lv, rv),
        HirBinOp::BitAnd => ctx.builder.ins().band(lv, rv),
        HirBinOp::BitOr => ctx.builder.ins().bor(lv, rv),
        HirBinOp::BitXor => ctx.builder.ins().bxor(lv, rv),
        HirBinOp::Shr => ctx.builder.ins().sshr(lv, rv),
        HirBinOp::And | HirBinOp::Or => unreachable!("handled by short-circuit"),
    })
}

fn compile_float_binop(ctx: &mut FuncCtx, op: HirBinOp, lv: Value, rv: Value) -> Option<Value> {
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
            let func_ref = ctx.get_runtime_func_ref("tok_pow_f64");
            let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
            ctx.builder.inst_results(call)[0]
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
    let lv = compile_expr(ctx, left).expect("codegen: left operand produced no value");
    let then_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();
    ctx.builder.append_block_param(merge_block, types::I8);

    // If left is falsy, short-circuit to false
    let cond = to_bool(ctx, lv, &left.ty);
    let false_val = ctx.builder.ins().iconst(types::I8, 0);
    ctx.builder
        .ins()
        .brif(cond, then_block, &[], merge_block, &[false_val]);

    ctx.builder.switch_to_block(then_block);
    ctx.builder.seal_block(then_block);
    let rv = compile_expr(ctx, right).expect("codegen: right operand produced no value");
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
    let lv = compile_expr(ctx, left).expect("codegen: left operand produced no value");
    let else_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();
    ctx.builder.append_block_param(merge_block, types::I8);

    let cond = to_bool(ctx, lv, &left.ty);
    let true_val = ctx.builder.ins().iconst(types::I8, 1);
    ctx.builder
        .ins()
        .brif(cond, merge_block, &[true_val], else_block, &[]);

    ctx.builder.switch_to_block(else_block);
    ctx.builder.seal_block(else_block);
    let rv = compile_expr(ctx, right).expect("codegen: right operand produced no value");
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
    let val = compile_expr(ctx, operand).expect("codegen: operand expr produced no value");
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

/// Look up a stdlib function's trampoline name and arity.
/// Returns `Some((trampoline_symbol, arity))` for known stdlib module functions.
fn get_stdlib_func(module: &str, field: &str) -> Option<(&'static str, usize)> {
    match (module, field) {
        // @"math" — 1-arg
        ("math", "sqrt") => Some(("tok_math_sqrt_t", 1)),
        ("math", "sin") => Some(("tok_math_sin_t", 1)),
        ("math", "cos") => Some(("tok_math_cos_t", 1)),
        ("math", "tan") => Some(("tok_math_tan_t", 1)),
        ("math", "asin") => Some(("tok_math_asin_t", 1)),
        ("math", "acos") => Some(("tok_math_acos_t", 1)),
        ("math", "atan") => Some(("tok_math_atan_t", 1)),
        ("math", "log") => Some(("tok_math_log_t", 1)),
        ("math", "log2") => Some(("tok_math_log2_t", 1)),
        ("math", "log10") => Some(("tok_math_log10_t", 1)),
        ("math", "exp") => Some(("tok_math_exp_t", 1)),
        ("math", "floor") => Some(("tok_math_floor_t", 1)),
        ("math", "ceil") => Some(("tok_math_ceil_t", 1)),
        ("math", "round") => Some(("tok_math_round_t", 1)),
        ("math", "abs") => Some(("tok_math_abs_t", 1)),
        // @"math" — 2-arg
        ("math", "pow") => Some(("tok_math_pow_t", 2)),
        ("math", "min") => Some(("tok_math_min_t", 2)),
        ("math", "max") => Some(("tok_math_max_t", 2)),
        ("math", "atan2") => Some(("tok_math_atan2_t", 2)),
        // @"math" — 0-arg
        ("math", "random") => Some(("tok_math_random_t", 0)),

        // @"str" — 1-arg
        ("str", "upper") => Some(("tok_str_upper_t", 1)),
        ("str", "lower") => Some(("tok_str_lower_t", 1)),
        ("str", "trim") => Some(("tok_str_trim_t", 1)),
        ("str", "trim_left") => Some(("tok_str_trim_left_t", 1)),
        ("str", "trim_right") => Some(("tok_str_trim_right_t", 1)),
        ("str", "chars") => Some(("tok_str_chars_t", 1)),
        ("str", "bytes") => Some(("tok_str_bytes_t", 1)),
        ("str", "rev") => Some(("tok_str_rev_t", 1)),
        ("str", "len") => Some(("tok_str_len_t", 1)),
        // @"str" — 2-arg
        ("str", "contains") => Some(("tok_str_contains_t", 2)),
        ("str", "starts_with") => Some(("tok_str_starts_with_t", 2)),
        ("str", "ends_with") => Some(("tok_str_ends_with_t", 2)),
        ("str", "index_of") => Some(("tok_str_index_of_t", 2)),
        ("str", "repeat") => Some(("tok_str_repeat_t", 2)),
        ("str", "split") => Some(("tok_str_split_t", 2)),
        // @"str" — 3-arg
        ("str", "replace") => Some(("tok_str_replace_t", 3)),
        ("str", "pad_left") => Some(("tok_str_pad_left_t", 3)),
        ("str", "pad_right") => Some(("tok_str_pad_right_t", 3)),
        ("str", "substr") => Some(("tok_str_substr_t", 3)),

        // @"json" — 1-arg
        ("json", "jparse") => Some(("tok_json_parse_t", 1)),
        ("json", "jstr") => Some(("tok_json_stringify_t", 1)),
        ("json", "jpretty") => Some(("tok_json_pretty_t", 1)),
        ("json", "parse") => Some(("tok_json_parse_t", 1)),
        ("json", "stringify") => Some(("tok_json_stringify_t", 1)),
        ("json", "pretty") => Some(("tok_json_pretty_t", 1)),

        // @"llm" — 1-arg
        ("llm", "ask") => Some(("tok_llm_ask_t", 1)),
        // @"llm" — 2-arg
        ("llm", "chat") => Some(("tok_llm_chat_2_t", 2)),

        // @"csv" — 1-arg
        ("csv", "cparse") => Some(("tok_csv_parse_t", 1)),
        ("csv", "cstr") => Some(("tok_csv_stringify_t", 1)),
        ("csv", "parse") => Some(("tok_csv_parse_t", 1)),
        ("csv", "stringify") => Some(("tok_csv_stringify_t", 1)),

        // @"tmpl" — 2-arg
        ("tmpl", "render") => Some(("tok_tmpl_render_t", 2)),
        ("tmpl", "apply") => Some(("tok_tmpl_apply_t", 2)),
        // @"tmpl" — 1-arg
        ("tmpl", "compile") => Some(("tok_tmpl_compile_t", 1)),

        // @"toon" — 1-arg
        ("toon", "tparse") => Some(("tok_toon_parse_t", 1)),
        ("toon", "tstr") => Some(("tok_toon_stringify_t", 1)),
        ("toon", "parse") => Some(("tok_toon_parse_t", 1)),
        ("toon", "stringify") => Some(("tok_toon_stringify_t", 1)),

        // @"os" — 0-arg
        ("os", "args") => Some(("tok_os_args_t", 0)),
        ("os", "cwd") => Some(("tok_os_cwd_t", 0)),
        ("os", "pid") => Some(("tok_os_pid_t", 0)),
        // @"os" — 1-arg
        ("os", "env") => Some(("tok_os_env_t", 1)),
        ("os", "exit") => Some(("tok_os_exit_t", 1)),
        ("os", "exec") => Some(("tok_os_exec_t", 1)),
        // @"os" — 2-arg
        ("os", "set_env") => Some(("tok_os_set_env_t", 2)),

        // @"io" — 0-arg
        ("io", "readall") => Some(("tok_io_readall_t", 0)),
        // @"io" — 1-arg (input with prompt)
        ("io", "input") => Some(("tok_io_input_1_t", 1)),

        // @"fs" — 1-arg
        ("fs", "fread") => Some(("tok_fs_fread_t", 1)),
        ("fs", "fexists") => Some(("tok_fs_fexists_t", 1)),
        ("fs", "fls") => Some(("tok_fs_fls_t", 1)),
        ("fs", "fmk") => Some(("tok_fs_fmk_t", 1)),
        ("fs", "frm") => Some(("tok_fs_frm_t", 1)),
        // @"fs" — 2-arg
        ("fs", "fwrite") => Some(("tok_fs_fwrite_t", 2)),
        ("fs", "fappend") => Some(("tok_fs_fappend_t", 2)),

        // @"http" — 1-arg
        ("http", "hget") => Some(("tok_http_hget_t", 1)),
        ("http", "hdel") => Some(("tok_http_hdel_t", 1)),
        // @"http" — 2-arg
        ("http", "hpost") => Some(("tok_http_hpost_t", 2)),
        ("http", "hput") => Some(("tok_http_hput_t", 2)),
        ("http", "serve") => Some(("tok_http_serve_t", 2)),

        // @"re" — 2-arg
        ("re", "rmatch") => Some(("tok_re_rmatch_t", 2)),
        ("re", "rfind") => Some(("tok_re_rfind_t", 2)),
        ("re", "rall") => Some(("tok_re_rall_t", 2)),
        // @"re" — 3-arg
        ("re", "rsub") => Some(("tok_re_rsub_t", 3)),

        // @"time" — 0-arg
        ("time", "now") => Some(("tok_time_now_t", 0)),
        // @"time" — 1-arg
        ("time", "sleep") => Some(("tok_time_sleep_t", 1)),
        // @"time" — 2-arg
        ("time", "fmt") => Some(("tok_time_fmt_t", 2)),

        _ => None,
    }
}

/// Look up a stdlib constant value (e.g. math.pi).
fn get_stdlib_const(module: &str, field: &str) -> Option<f64> {
    match (module, field) {
        ("math", "pi") => Some(std::f64::consts::PI),
        ("math", "e") => Some(std::f64::consts::E),
        ("math", "inf") => Some(f64::INFINITY),
        ("math", "nan") => Some(f64::NAN),
        _ => None,
    }
}

// ─── Builtin call helpers ─────────────────────────────────────────────

/// Compile a 1-arg builtin: compile arg → unwrap_any_ptr → call runtime → return ptr
fn compile_builtin_1_ptr(ctx: &mut FuncCtx, arg: &HirExpr, runtime_fn: &str) -> Option<Value> {
    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
    let ptr = unwrap_any_ptr(ctx, val, &arg.ty);
    let func_ref = ctx.get_runtime_func_ref(runtime_fn);
    let call = ctx.builder.ins().call(func_ref, &[ptr]);
    Some(ctx.builder.inst_results(call)[0])
}

/// Compile a 1-arg builtin that returns (tag, data): compile arg → unwrap → call → from_tokvalue
fn compile_builtin_1_tokvalue(
    ctx: &mut FuncCtx,
    arg: &HirExpr,
    runtime_fn: &str,
    result_ty: &Type,
) -> Option<Value> {
    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
    let ptr = unwrap_any_ptr(ctx, val, &arg.ty);
    let func_ref = ctx.get_runtime_func_ref(runtime_fn);
    let call = ctx.builder.ins().call(func_ref, &[ptr]);
    let results = ctx.builder.inst_results(call);
    Some(from_tokvalue(ctx, results[0], results[1], result_ty))
}

/// Compile a 2-arg builtin: compile both → unwrap → call → return ptr
fn compile_builtin_2_ptr(ctx: &mut FuncCtx, args: &[HirExpr], runtime_fn: &str) -> Option<Value> {
    let a_raw = compile_expr(ctx, &args[0]).expect("codegen: args[0] produced no value");
    let a = unwrap_any_ptr(ctx, a_raw, &args[0].ty);
    let b_raw = compile_expr(ctx, &args[1]).expect("codegen: args[1] produced no value");
    let b = unwrap_any_ptr(ctx, b_raw, &args[1].ty);
    let func_ref = ctx.get_runtime_func_ref(runtime_fn);
    let call = ctx.builder.ins().call(func_ref, &[a, b]);
    Some(ctx.builder.inst_results(call)[0])
}

fn compile_call(
    ctx: &mut FuncCtx,
    func_expr: &HirExpr,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    // ── Stdlib direct-call optimization ─────────────────────────
    // Detect m.func(args) where m is a known stdlib import.
    // Emit a direct call to the trampoline, bypassing map lookup + indirect dispatch.
    if let HirExprKind::Member { target, field } = &func_expr.kind {
        if let HirExprKind::Ident(var_name) = &target.kind {
            if let Some(module_name) = ctx.stdlib_imports.get(var_name.as_str()).cloned() {
                if let Some((trampoline, _arity)) = get_stdlib_func(&module_name, field) {
                    let func_ref = ctx.get_runtime_func_ref(trampoline);
                    let null_env = ctx.builder.ins().iconst(PTR, 0i64);
                    let mut call_args = vec![null_env];
                    for arg in args {
                        let v = compile_expr(ctx, arg)
                            .unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
                        let (tag, data) = to_tokvalue(ctx, v, &arg.ty);
                        call_args.push(tag);
                        call_args.push(data);
                    }
                    let call = ctx.builder.ins().call(func_ref, &call_args);
                    let results = ctx.builder.inst_results(call);
                    return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
                }
            }
        }
    }

    // Check if this is a call to a known function name
    if let HirExprKind::Ident(name) = &func_expr.kind {
        // User-defined functions take priority over builtins
        if ctx.compiler.declared_funcs.contains_key(name.as_str()) {
            // Try inlining small user functions (single-expression body, non-recursive)
            if can_inline_user_func(ctx, name) {
                if let Some(result) = compile_inline_user_func(ctx, name, args, result_ty) {
                    return Some(result);
                }
            }
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
                    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
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
                    let arr_raw =
                        compile_expr(ctx, &args[0]).expect("codegen: args[0] produced no value");
                    let mut arr = unwrap_any_ptr(ctx, arr_raw, &args[0].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_array_push");
                    for arg in &args[1..] {
                        let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
                        let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                        let call = ctx.builder.ins().call(func_ref, &[arr, tag, data]);
                        arr = ctx.builder.inst_results(call)[0];
                    }
                    return Some(arr);
                }
            }
            "sort" => {
                if args.len() == 1 {
                    return compile_builtin_1_ptr(ctx, &args[0], "tok_array_sort");
                }
            }
            "rev" => {
                if args.len() == 1 {
                    return compile_builtin_1_ptr(ctx, &args[0], "tok_array_rev");
                }
            }
            "flat" => {
                if args.len() == 1 {
                    return compile_builtin_1_ptr(ctx, &args[0], "tok_array_flat");
                }
            }
            "uniq" => {
                if args.len() == 1 {
                    return compile_builtin_1_ptr(ctx, &args[0], "tok_array_uniq");
                }
            }
            "join" => {
                if args.len() >= 2 {
                    return compile_builtin_2_ptr(ctx, args, "tok_array_join");
                }
            }
            "split" => {
                if args.len() >= 2 {
                    return compile_builtin_2_ptr(ctx, args, "tok_string_split");
                }
            }
            "trim" => {
                if args.len() == 1 {
                    return compile_builtin_1_ptr(ctx, &args[0], "tok_string_trim");
                }
            }
            "keys" => {
                if args.len() == 1 {
                    return compile_builtin_1_ptr(ctx, &args[0], "tok_map_keys");
                }
            }
            "vals" => {
                if args.len() == 1 {
                    return compile_builtin_1_ptr(ctx, &args[0], "tok_map_vals");
                }
            }
            "has" => {
                if args.len() >= 2 {
                    return compile_builtin_2_ptr(ctx, args, "tok_map_has");
                }
            }
            "del" => {
                if args.len() >= 2 {
                    return compile_builtin_2_ptr(ctx, args, "tok_map_del");
                }
            }
            "int" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
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
                    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
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
                    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
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
                    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
                    if matches!(arg.ty, Type::Any) {
                        // Fully dynamic: dispatch at runtime
                        let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                        let func_ref = ctx.get_runtime_func_ref("tok_value_abs");
                        let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                        let results = ctx.builder.inst_results(call);
                        return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
                    }
                    let is_float = matches!(arg.ty, Type::Float);
                    let func_name = if is_float {
                        "tok_abs_float"
                    } else {
                        "tok_abs_int"
                    };
                    let func_ref = ctx.get_runtime_func_ref(func_name);
                    let call = ctx.builder.ins().call(func_ref, &[val]);
                    let raw = ctx.builder.inst_results(call)[0];
                    // Wrap if caller expects Any
                    if matches!(result_ty, Type::Any) {
                        let tag_val = ctx
                            .builder
                            .ins()
                            .iconst(types::I64, if is_float { TAG_FLOAT } else { TAG_INT });
                        let data_val = if is_float {
                            ctx.builder.ins().bitcast(types::I64, MemFlags::new(), raw)
                        } else {
                            raw
                        };
                        return Some(alloc_tokvalue_on_stack(ctx, tag_val, data_val));
                    }
                    return Some(raw);
                }
            }
            "floor" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
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
                        let tag_val = ctx.builder.ins().iconst(types::I64, TAG_INT);
                        return Some(alloc_tokvalue_on_stack(ctx, tag_val, raw));
                    }
                    return Some(raw);
                }
            }
            "ceil" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
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
                        let tag_val = ctx.builder.ins().iconst(types::I64, TAG_INT);
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
                    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
                    let func_ref = ctx.get_runtime_func_ref("tok_exit");
                    ctx.builder.ins().call(func_ref, &[val]);
                }
                return None;
            }
            "chan" => {
                let cap = if let Some(arg) = args.first() {
                    compile_expr(ctx, arg).expect("codegen: arg produced no value")
                } else {
                    ctx.builder.ins().iconst(types::I64, 0)
                };
                let func_ref = ctx.get_runtime_func_ref("tok_channel_alloc");
                let call = ctx.builder.ins().call(func_ref, &[cap]);
                return Some(ctx.builder.inst_results(call)[0]);
            }
            "type" => {
                if let Some(arg) = args.first() {
                    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
                    let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_type_of");
                    let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "min" => {
                if args.len() == 1 {
                    return compile_builtin_1_tokvalue(ctx, &args[0], "tok_array_min", result_ty);
                }
            }
            "max" => {
                if args.len() == 1 {
                    return compile_builtin_1_tokvalue(ctx, &args[0], "tok_array_max", result_ty);
                }
            }
            "sum" => {
                if args.len() == 1 {
                    return compile_builtin_1_tokvalue(ctx, &args[0], "tok_array_sum", result_ty);
                }
            }
            "slice" => {
                if args.len() >= 3 {
                    let target_raw =
                        compile_expr(ctx, &args[0]).expect("codegen: args[0] produced no value");
                    let start =
                        compile_expr(ctx, &args[1]).expect("codegen: args[1] produced no value");
                    let end =
                        compile_expr(ctx, &args[2]).expect("codegen: args[2] produced no value");
                    if matches!(args[0].ty, Type::Any) {
                        // Fully dynamic: dispatch at runtime by tag
                        let (tag, data) = to_tokvalue(ctx, target_raw, &args[0].ty);
                        let func_ref = ctx.get_runtime_func_ref("tok_value_slice");
                        let call = ctx.builder.ins().call(func_ref, &[tag, data, start, end]);
                        let results = ctx.builder.inst_results(call);
                        return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
                    }
                    let target = unwrap_any_ptr(ctx, target_raw, &args[0].ty);
                    let (func_name, tag_const) = match &args[0].ty {
                        Type::Array(_) => ("tok_array_slice", TAG_ARRAY),
                        Type::Str => ("tok_string_slice", TAG_STRING),
                        _ => return None,
                    };
                    let func_ref = ctx.get_runtime_func_ref(func_name);
                    let call = ctx.builder.ins().call(func_ref, &[target, start, end]);
                    let raw = ctx.builder.inst_results(call)[0];
                    // Wrap as TokValue if caller expects Any
                    if matches!(result_ty, Type::Any) {
                        let tag = ctx.builder.ins().iconst(types::I64, tag_const);
                        return Some(alloc_tokvalue_on_stack(ctx, tag, raw));
                    }
                    return Some(raw);
                }
            }
            "pmap" => {
                if args.len() >= 2 {
                    let arr_raw =
                        compile_expr(ctx, &args[0]).expect("codegen: args[0] produced no value");
                    let arr = unwrap_any_ptr(ctx, arr_raw, &args[0].ty);
                    let closure =
                        compile_expr(ctx, &args[1]).expect("codegen: args[1] produced no value");
                    let closure_ptr = unwrap_any_ptr(ctx, closure, &args[1].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_pmap");
                    let call = ctx.builder.ins().call(func_ref, &[arr, closure_ptr]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "is" => {
                if args.len() >= 2 {
                    // compile_expr returns None for Nil, so handle that
                    let val_opt = compile_expr(ctx, &args[0]);
                    let (tag, data) = if let Some(val) = val_opt {
                        to_tokvalue(ctx, val, &args[0].ty)
                    } else {
                        // Nil value
                        let tag = ctx.builder.ins().iconst(types::I64, 0); // TAG_NIL
                        let data = ctx.builder.ins().iconst(types::I64, 0);
                        (tag, data)
                    };
                    let type_str =
                        compile_expr(ctx, &args[1]).expect("codegen: args[1] produced no value");
                    let str_ptr = unwrap_any_ptr(ctx, type_str, &args[1].ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_is");
                    let call = ctx.builder.ins().call(func_ref, &[tag, data, str_ptr]);
                    return Some(ctx.builder.inst_results(call)[0]);
                }
            }
            "pop" => {
                if args.len() == 1 {
                    return compile_builtin_1_tokvalue(ctx, &args[0], "tok_array_pop", result_ty);
                }
            }
            "freq" => {
                if args.len() == 1 {
                    return compile_builtin_1_ptr(ctx, &args[0], "tok_array_freq");
                }
            }
            "zip" => {
                if args.len() >= 2 {
                    return compile_builtin_2_ptr(ctx, args, "tok_array_zip");
                }
            }
            "top" => {
                if args.len() >= 2 {
                    return compile_builtin_2_ptr(ctx, args, "tok_map_top");
                }
            }
            "args" => {
                let func_ref = ctx.get_runtime_func_ref("tok_args");
                let call = ctx.builder.ins().call(func_ref, &[]);
                return Some(ctx.builder.inst_results(call)[0]);
            }
            "env" => {
                if args.len() == 1 {
                    return compile_builtin_1_tokvalue(ctx, &args[0], "tok_env", result_ty);
                }
            }
            _ => {
                // Direct call optimization: if we know the FuncId, skip indirect dispatch
                if let Some(kc) = ctx.known_closures.get(name).cloned() {
                    // Try specialized call if all arg types are concrete
                    let arg_types: Vec<Type> = args.iter().map(|a| a.ty.clone()).collect();
                    let all_concrete = arg_types
                        .iter()
                        .all(|t| matches!(t, Type::Int | Type::Float | Type::Bool));
                    if all_concrete {
                        // Try inlining first for simple lambdas (single-expression body)
                        if can_inline_closure_call(
                            &ctx.compiler.pending_lambdas[kc.pending_idx],
                            &arg_types,
                            name,
                        ) {
                            return compile_inline_closure_call(
                                ctx, name, &kc, args, &arg_types, result_ty,
                            );
                        }
                        return compile_specialized_closure_call(
                            ctx, name, &kc, args, &arg_types, result_ty,
                        );
                    }
                    return compile_direct_closure_call(
                        ctx, kc.func_id, kc.env_ptr, args, result_ty,
                    );
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
                        let closure_ptr =
                            ctx.builder
                                .ins()
                                .load(types::I64, MemFlags::trusted(), tokval_ptr, 8);
                        return compile_closure_call(ctx, closure_ptr, args, result_ty);
                    }
                }
                // Unknown — fall through to generic call below
            }
        }
    }

    // Generic function call (through closure expression)
    let func_val = compile_expr(ctx, func_expr);
    if let Some(raw_val) = func_val {
        // If func expr is Any-typed, extract closure ptr from TokValue data field
        let closure_ptr = if matches!(
            &func_expr.ty,
            Type::Any | Type::Optional(_) | Type::Result(_)
        ) {
            ctx.builder
                .ins()
                .load(types::I64, MemFlags::trusted(), raw_val, 8)
        } else {
            raw_val
        };
        return compile_closure_call(ctx, closure_ptr, args, result_ty);
    }

    Some(ctx.builder.ins().iconst(types::I64, 0))
}

/// Check if a user-defined function can be inlined at the call site.
/// Eligible: single-expression or single-return body, non-recursive, small.
fn can_inline_user_func(ctx: &FuncCtx, name: &str) -> bool {
    let (params, _ret_type, body) = match ctx.compiler.func_bodies.get(name) {
        Some(v) => v,
        None => return false,
    };
    // Must be a single statement
    if body.len() != 1 {
        return false;
    }
    // All parameters must be scalar types or known closures
    // (skip inlining for Tuple/Map/Array params which have complex ABI)
    if !params
        .iter()
        .all(|p| matches!(p.ty, Type::Int | Type::Float | Type::Bool | Type::Any))
    {
        return false;
    }
    let expr = match &body[0] {
        HirStmt::Expr(e) => e,
        HirStmt::Return(Some(e)) => e,
        _ => return false,
    };
    // Don't inline self-recursive functions
    if contains_self_call(expr, name) {
        return false;
    }
    // Don't inline functions that contain embedded Return statements
    // (e.g., from ?^ error propagation or cond?^expr)
    // because Returns would jump to the wrong return block when inlined.
    if expr_contains_return(expr) {
        return false;
    }
    true
}

/// Check if an HIR expression tree contains any Return statements
/// (nested in If/Block/Loop etc.)
fn expr_contains_return(expr: &HirExpr) -> bool {
    match &expr.kind {
        HirExprKind::If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            expr_contains_return(cond)
                || then_body.iter().any(stmt_contains_return)
                || then_expr.as_ref().is_some_and(|e| expr_contains_return(e))
                || else_body.iter().any(stmt_contains_return)
                || else_expr.as_ref().is_some_and(|e| expr_contains_return(e))
        }
        HirExprKind::Block { stmts, expr: e } => {
            stmts.iter().any(stmt_contains_return)
                || e.as_ref().is_some_and(|e| expr_contains_return(e))
        }
        HirExprKind::BinOp { left, right, .. } => {
            expr_contains_return(left) || expr_contains_return(right)
        }
        HirExprKind::UnaryOp { operand, .. } => expr_contains_return(operand),
        HirExprKind::Call { func, args } => {
            expr_contains_return(func) || args.iter().any(expr_contains_return)
        }
        HirExprKind::Index { target, index } => {
            expr_contains_return(target) || expr_contains_return(index)
        }
        HirExprKind::Member { target, .. } => expr_contains_return(target),
        HirExprKind::Loop { body, .. } => body.iter().any(stmt_contains_return),
        HirExprKind::Array(elems) => elems.iter().any(expr_contains_return),
        HirExprKind::Tuple(elems) => elems.iter().any(expr_contains_return),
        _ => false,
    }
}

fn stmt_contains_return(stmt: &HirStmt) -> bool {
    match stmt {
        HirStmt::Return(_) => true,
        HirStmt::Expr(e) => expr_contains_return(e),
        HirStmt::Assign { value, .. } => expr_contains_return(value),
        _ => false,
    }
}

/// Inline a user-defined function at the call site.
fn compile_inline_user_func(
    ctx: &mut FuncCtx,
    name: &str,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    let (params, _ret_type, body) = ctx.compiler.func_bodies.get(name).cloned()?;
    if params.len() != args.len() {
        return None;
    }

    // Compile argument expressions first
    let mut arg_vals = Vec::new();
    let mut arg_types = Vec::new();
    for arg in args {
        let v = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        arg_vals.push(v);
        arg_types.push(arg.ty.clone());
    }

    // Save old bindings and bind function parameters to arg values.
    // Also propagate known closures from call site to inlined body.
    let mut old_bindings: Vec<(String, Option<(Variable, Type)>)> = Vec::new();
    let mut old_kc_bindings: Vec<(String, Option<KnownClosure>)> = Vec::new();
    for (i, param) in params.iter().enumerate() {
        old_bindings.push((param.name.clone(), ctx.vars.remove(&param.name)));
        // If param is Any and arg is concrete, keep as concrete type
        let actual_ty = if matches!(param.ty, Type::Any) {
            &arg_types[i]
        } else {
            &param.ty
        };
        let ct = cl_type_or_i64(actual_ty);
        let var = ctx.new_var(ct);
        // Coerce arg value to the parameter type if needed
        let coerced = if matches!(param.ty, Type::Any) && !matches!(arg_types[i], Type::Any) {
            // Parameter is Any but we keep concrete type (no boxing needed when inlining)
            arg_vals[i]
        } else {
            coerce_value(ctx, arg_vals[i], &arg_types[i], actual_ty)
        };
        ctx.builder.def_var(var, coerced);
        ctx.vars
            .insert(param.name.clone(), (var, actual_ty.clone()));

        // Propagate known closure info: if the argument is a known closure variable,
        // bind that closure info to the parameter name so fn(x) can be inlined/specialized
        if let HirExprKind::Ident(arg_name) = &args[i].kind {
            if let Some(kc) = ctx.known_closures.get(arg_name).cloned() {
                old_kc_bindings.push((param.name.clone(), ctx.known_closures.remove(&param.name)));
                ctx.known_closures.insert(param.name.clone(), kc);
            }
        }
    }

    // Retype the body with concrete arg types
    let mut type_map = HashMap::new();
    for (i, param) in params.iter().enumerate() {
        let actual_ty = if matches!(param.ty, Type::Any) {
            &arg_types[i]
        } else {
            &param.ty
        };
        type_map.insert(param.name.clone(), actual_ty.clone());
    }
    let retyped = retype_body(&body, &type_map);
    let retyped = unwrap_return_stmts(retyped);

    // Determine result type from retyped body
    let body_result_ty = retyped
        .last()
        .and_then(|s| match s {
            HirStmt::Expr(e) => Some(e.ty.clone()),
            _ => None,
        })
        .unwrap_or(Type::Nil);

    // Compile the body inline
    let body_result = compile_body(ctx, &retyped, &body_result_ty);

    // Restore old bindings
    for (pname, old) in old_bindings {
        ctx.vars.remove(&pname);
        if let Some(old_val) = old {
            ctx.vars.insert(pname, old_val);
        }
    }
    // Restore old known_closure bindings
    for (pname, old) in old_kc_bindings {
        ctx.known_closures.remove(&pname);
        if let Some(old_val) = old {
            ctx.known_closures.insert(pname, old_val);
        }
    }

    // Coerce result to caller's expected type
    if let Some(val) = body_result {
        if matches!(result_ty, Type::Any | Type::Optional(_) | Type::Result(_))
            && !matches!(body_result_ty, Type::Any)
        {
            let (tag, data) = to_tokvalue(ctx, val, &body_result_ty);
            Some(alloc_tokvalue_on_stack(ctx, tag, data))
        } else {
            Some(coerce_value(ctx, val, &body_result_ty, result_ty))
        }
    } else {
        None
    }
}

fn compile_user_func_call(
    ctx: &mut FuncCtx,
    name: &str,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    // TCO: if this is a tail call to self, compile args and jump to loop header
    if ctx.tco_func_name.as_deref() == Some(name) {
        if let Some(loop_header) = ctx.tco_loop_header {
            let func_sig = ctx.compiler.func_sigs.get(name).cloned();
            let mut jump_vals = Vec::new();
            for (i, arg) in args.iter().enumerate() {
                if let Some(v) = compile_expr(ctx, arg) {
                    let param_ty = func_sig
                        .as_ref()
                        .and_then(|s| s.0.get(i))
                        .cloned()
                        .unwrap_or(arg.ty.clone());
                    if matches!(param_ty, Type::Any) {
                        let (tag, data) = to_tokvalue(ctx, v, &arg.ty);
                        jump_vals.push(tag);
                        jump_vals.push(data);
                    } else if matches!(arg.ty, Type::Any) {
                        let coerced = coerce_value(ctx, v, &arg.ty, &param_ty);
                        jump_vals.push(coerced);
                    } else {
                        jump_vals.push(v);
                    }
                }
            }
            ctx.builder.ins().jump(loop_header, &jump_vals);
            // Create a dead block for any unreachable code after the tail call
            let dead_block = ctx.builder.create_block();
            ctx.builder.switch_to_block(dead_block);
            ctx.builder.seal_block(dead_block);
            ctx.block_terminated = true;
            return None;
        }
    }

    let func_sig = ctx.compiler.func_sigs.get(name).cloned();
    let mut arg_vals = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        if let Some(v) = compile_expr(ctx, arg) {
            let param_ty = func_sig
                .as_ref()
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
    let ret_ty = func_sig
        .as_ref()
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
    let fn_ptr = ctx
        .builder
        .ins()
        .load(PTR, MemFlags::trusted(), closure_ptr, 8);
    let env_ptr = ctx
        .builder
        .ins()
        .load(PTR, MemFlags::trusted(), closure_ptr, 16);

    // Build or reuse cached signature for indirect call: (env: PTR, tag0: I64, data0: I64, ...) -> (I64, I64)
    let n_args = args.len();
    let sig_ref = if let Some(&cached) = ctx.closure_sig_cache.get(&n_args) {
        cached
    } else {
        let mut sig = ctx.compiler.module.make_signature();
        sig.params.push(AbiParam::new(PTR)); // env
        for _ in 0..n_args {
            sig.params.push(AbiParam::new(types::I64)); // tag
            sig.params.push(AbiParam::new(types::I64)); // data
        }
        sig.returns.push(AbiParam::new(types::I64)); // ret tag
        sig.returns.push(AbiParam::new(types::I64)); // ret data
        let sr = ctx.builder.import_signature(sig);
        ctx.closure_sig_cache.insert(n_args, sr);
        sr
    };

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
    let func_ref = ctx
        .compiler
        .module
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
        if stypes == arg_types {
            Some((sid, sret.clone()))
        } else {
            None
        }
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
        let ret_type = retyped_body
            .last()
            .and_then(|s| match s {
                HirStmt::Expr(e) => {
                    if matches!(e.ty, Type::Any) {
                        None
                    } else {
                        Some(e.ty.clone())
                    }
                }
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

        let func_id = ctx
            .compiler
            .module
            .declare_function(&spec_name, Linkage::Local, &sig)
            .expect("codegen: failed to declare specialized lambda");

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
    let func_ref = ctx
        .compiler
        .module
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

// ─── Inline closure calls ──────────────────────────────────────────────

/// Check if a known closure can be inlined at its call site.
/// Returns true if the lambda body is a single expression and all types are concrete.
fn can_inline_closure_call(pending: &PendingLambda, arg_types: &[Type], var_name: &str) -> bool {
    // All args must be concrete
    if !arg_types
        .iter()
        .all(|t| matches!(t, Type::Int | Type::Float | Type::Bool))
    {
        return false;
    }
    // Body must be exactly one statement: Expr(e) or Return(Some(e))
    if pending.body.len() != 1 {
        return false;
    }
    let body_expr = match &pending.body[0] {
        HirStmt::Expr(e) => e,
        HirStmt::Return(Some(e)) => e,
        _ => return false,
    };
    // All captures must be concrete
    if !pending
        .captures
        .iter()
        .all(|c| matches!(c.ty, Type::Int | Type::Float | Type::Bool))
    {
        return false;
    }
    // Don't inline self-recursive calls
    !contains_self_call(body_expr, var_name)
}

/// Check if an HIR statement contains a call to a function with the given name.
fn stmt_contains_self_call(stmt: &HirStmt, name: &str) -> bool {
    match stmt {
        HirStmt::Expr(e) => contains_self_call(e, name),
        HirStmt::Return(Some(e)) => contains_self_call(e, name),
        HirStmt::Assign { value, .. } => contains_self_call(value, name),
        _ => false,
    }
}

/// Check if an HIR expression contains a call to a function with the given name.
fn contains_self_call(expr: &HirExpr, name: &str) -> bool {
    match &expr.kind {
        HirExprKind::Call { func, args } => {
            if let HirExprKind::Ident(callee) = &func.kind {
                if callee == name {
                    return true;
                }
            }
            contains_self_call(func, name) || args.iter().any(|a| contains_self_call(a, name))
        }
        HirExprKind::BinOp { left, right, .. } => {
            contains_self_call(left, name) || contains_self_call(right, name)
        }
        HirExprKind::UnaryOp { operand, .. } => contains_self_call(operand, name),
        HirExprKind::If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            contains_self_call(cond, name)
                || then_body.iter().any(|s| stmt_contains_self_call(s, name))
                || then_expr
                    .as_ref()
                    .is_some_and(|e| contains_self_call(e, name))
                || else_body.iter().any(|s| stmt_contains_self_call(s, name))
                || else_expr
                    .as_ref()
                    .is_some_and(|e| contains_self_call(e, name))
        }
        HirExprKind::Index { target, index } => {
            contains_self_call(target, name) || contains_self_call(index, name)
        }
        HirExprKind::Member { target, .. } => contains_self_call(target, name),
        _ => false,
    }
}

/// Check if a function body ends with a self-tail-call (directly or in both branches of an if).
fn is_self_tail_recursive(body: &[HirStmt], name: &str) -> bool {
    let last = match body.last() {
        Some(s) => s,
        None => return false,
    };
    match last {
        HirStmt::Expr(e) => is_tail_call_expr(e, name),
        HirStmt::Return(Some(e)) => is_tail_call_expr(e, name),
        _ => false,
    }
}

/// Check if an expression is a tail call to `name` (direct call or if-then-else where at least
/// one branch is a tail call).
fn is_tail_call_expr(expr: &HirExpr, name: &str) -> bool {
    match &expr.kind {
        HirExprKind::Call { func, args: _ } => {
            if let HirExprKind::Ident(callee) = &func.kind {
                callee == name
            } else {
                false
            }
        }
        HirExprKind::If {
            then_body: _,
            then_expr,
            else_body: _,
            else_expr,
            ..
        } => {
            // At least one branch must be a tail call for TCO to be useful.
            // We only transform the branches that ARE tail calls; non-tail branches
            // return normally.
            let then_tail = then_expr
                .as_ref()
                .is_some_and(|e| is_tail_call_expr(e, name));
            let else_tail = else_expr
                .as_ref()
                .is_some_and(|e| is_tail_call_expr(e, name));
            then_tail || else_tail
        }
        _ => false,
    }
}

/// Inline a known lambda call at the call site.
/// Instead of emitting a function call, we compile the lambda body directly
/// into the caller's instruction stream.
fn compile_inline_closure_call(
    ctx: &mut FuncCtx,
    _name: &str,
    kc: &KnownClosure,
    args: &[HirExpr],
    arg_types: &[Type],
    result_ty: &Type,
) -> Option<Value> {
    // Get the pending lambda's body and metadata
    let params = ctx.compiler.pending_lambdas[kc.pending_idx].params.clone();
    let captures = ctx.compiler.pending_lambdas[kc.pending_idx]
        .captures
        .clone();
    let body = ctx.compiler.pending_lambdas[kc.pending_idx].body.clone();

    // Compile argument expressions before binding anything
    let mut arg_vals = Vec::new();
    for arg in args {
        let v = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        arg_vals.push(v);
    }

    // Save old bindings and bind lambda parameters to compiled arg values
    let mut old_param_bindings: Vec<(String, Option<(Variable, Type)>)> = Vec::new();
    for (i, param) in params.iter().enumerate() {
        old_param_bindings.push((param.name.clone(), ctx.vars.remove(&param.name)));
        let ct = cl_type_or_i64(&arg_types[i]);
        let var = ctx.new_var(ct);
        ctx.builder.def_var(var, arg_vals[i]);
        ctx.vars
            .insert(param.name.clone(), (var, arg_types[i].clone()));
    }

    // Handle captures: load from env_ptr to preserve snapshot semantics
    let mut old_capture_bindings: Vec<(String, Option<(Variable, Type)>)> = Vec::new();
    if !captures.is_empty() {
        let env_ptr = kc.env_ptr;
        for (i, cap) in captures.iter().enumerate() {
            old_capture_bindings.push((cap.name.clone(), ctx.vars.remove(&cap.name)));
            let offset = (i * 16) as i32;
            match &cap.ty {
                Type::Int => {
                    let data = ctx.builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        env_ptr,
                        offset + 8,
                    );
                    let var = ctx.new_var(types::I64);
                    ctx.builder.def_var(var, data);
                    ctx.vars.insert(cap.name.clone(), (var, Type::Int));
                }
                Type::Float => {
                    let data = ctx.builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        env_ptr,
                        offset + 8,
                    );
                    let fval = ctx.builder.ins().bitcast(types::F64, MemFlags::new(), data);
                    let var = ctx.new_var(types::F64);
                    ctx.builder.def_var(var, fval);
                    ctx.vars.insert(cap.name.clone(), (var, Type::Float));
                }
                Type::Bool => {
                    let data = ctx.builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        env_ptr,
                        offset + 8,
                    );
                    let bval = ctx.builder.ins().ireduce(types::I8, data);
                    let var = ctx.new_var(types::I8);
                    ctx.builder.def_var(var, bval);
                    ctx.vars.insert(cap.name.clone(), (var, Type::Bool));
                }
                _ => unreachable!("can_inline_closure_call requires all captures concrete"),
            }
        }
    }

    // Build type map and retype body for concrete types
    let mut type_map = HashMap::new();
    for (param, arg_ty) in params.iter().zip(arg_types.iter()) {
        type_map.insert(param.name.clone(), arg_ty.clone());
    }
    for cap in &captures {
        type_map.insert(cap.name.clone(), cap.ty.clone());
    }
    let retyped = retype_body(&body, &type_map);
    let retyped = unwrap_return_stmts(retyped);

    // Determine result type from the retyped body
    let body_result_ty = retyped
        .last()
        .and_then(|s| match s {
            HirStmt::Expr(e) => Some(e.ty.clone()),
            _ => None,
        })
        .unwrap_or(Type::Int);

    // Compile the retyped body inline
    let body_result = compile_body(ctx, &retyped, &body_result_ty);

    // Restore old bindings for parameters
    for (pname, old) in old_param_bindings {
        ctx.vars.remove(&pname);
        if let Some(old_val) = old {
            ctx.vars.insert(pname, old_val);
        }
    }
    // Restore old bindings for captures
    for (cname, old) in old_capture_bindings {
        ctx.vars.remove(&cname);
        if let Some(old_val) = old {
            ctx.vars.insert(cname, old_val);
        }
    }

    // Coerce result to caller's expected type
    if let Some(val) = body_result {
        if matches!(result_ty, Type::Any | Type::Optional(_) | Type::Result(_))
            && !matches!(body_result_ty, Type::Any)
        {
            let (tag, data) = to_tokvalue(ctx, val, &body_result_ty);
            Some(alloc_tokvalue_on_stack(ctx, tag, data))
        } else {
            Some(coerce_value(ctx, val, &body_result_ty, result_ty))
        }
    } else {
        None
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
    let HirExprKind::Lambda { params, body, .. } = &lambda_expr.kind else {
        unreachable!()
    };
    let elem_type = match &arr_expr.ty {
        Type::Array(inner) => inner.as_ref().clone(),
        _ => Type::Any,
    };
    let param_name = &params[0].name;

    // Compile source array
    let arr_raw = compile_expr(ctx, arr_expr).expect("codegen: array expr produced no value");
    let arr = if matches!(
        &arr_expr.ty,
        Type::Any | Type::Optional(_) | Type::Result(_)
    ) {
        ctx.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), arr_raw, 8)
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
    ctx.vars
        .insert(param_name.clone(), (elem_var, elem_type.clone()));

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
    ctx.builder
        .ins()
        .brif(cond, body_block, &[], exit_block, &[]);

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
            let pred_ty = retyped
                .last()
                .and_then(|s| match s {
                    HirStmt::Expr(e) => Some(e.ty.clone()),
                    _ => None,
                })
                .unwrap_or(Type::Bool);
            let bool_val = to_bool(ctx, pred_val, &pred_ty);

            ctx.builder
                .ins()
                .brif(bool_val, push_block, &[], inc_block, &[]);
        }

        // Push block: add element to result array
        ctx.builder.switch_to_block(push_block);
        ctx.builder.seal_block(push_block);
        let push_ref = ctx.get_runtime_func_ref("tok_array_push");
        ctx.builder
            .ins()
            .call(push_ref, &[result_arr, elem_tag, elem_data]);
        ctx.builder.ins().jump(inc_block, &[]);
    } else if !ctx.block_terminated {
        ctx.builder.ins().jump(inc_block, &[]);
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
        let tag = ctx.builder.ins().iconst(types::I64, TAG_ARRAY);
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
    let HirExprKind::Lambda { params, body, .. } = &lambda_expr.kind else {
        unreachable!()
    };
    let elem_type = match &arr_expr.ty {
        Type::Array(inner) => inner.as_ref().clone(),
        _ => Type::Any,
    };
    let acc_name = &params[0].name;
    let elem_name = &params[1].name;

    // Compile source array
    let arr_raw = compile_expr(ctx, arr_expr).expect("codegen: array expr produced no value");
    let arr = if matches!(
        &arr_expr.ty,
        Type::Any | Type::Optional(_) | Type::Result(_)
    ) {
        ctx.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), arr_raw, 8)
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
        let iv = compile_expr(ctx, init_expr).expect("codegen: init expr produced no value");
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
    ctx.vars
        .insert(acc_name.clone(), (acc_var, acc_type.clone()));
    ctx.vars
        .insert(elem_name.clone(), (elem_var, elem_type.clone()));

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
    ctx.builder
        .ins()
        .brif(cond, body_block, &[], exit_block, &[]);

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
        let body_ty = retyped
            .last()
            .and_then(|s| match s {
                HirStmt::Expr(e) => Some(e.ty.clone()),
                _ => None,
            })
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
        let val = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        let use_newline = newline && i == args.len() - 1;
        let func_name = match &arg.ty {
            Type::Int => {
                if use_newline {
                    "tok_println_int"
                } else {
                    "tok_print_int"
                }
            }
            Type::Float => {
                if use_newline {
                    "tok_println_float"
                } else {
                    "tok_print_float"
                }
            }
            Type::Str => {
                if use_newline {
                    "tok_println_string"
                } else {
                    "tok_print_string"
                }
            }
            Type::Bool => {
                if use_newline {
                    "tok_println_bool"
                } else {
                    "tok_print_bool"
                }
            }
            _ => {
                // Pack as TokValue
                let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                let func_name = if use_newline {
                    "tok_println"
                } else {
                    "tok_print"
                };
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
    let cond_val = compile_expr(ctx, cond).expect("codegen: condition expr produced no value");
    let cond_bool = to_bool(ctx, cond_val, &cond.ty);

    let then_block = ctx.builder.create_block();
    let else_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();

    // If either branch is Any but result_ty is concrete, we must use Any semantics
    // internally because the Any branch might hold a different runtime type.
    let then_ty = then_expr.as_ref().map(|e| &e.ty);
    let else_ty = else_expr.as_ref().map(|e| &e.ty);
    let any_branch = then_ty.is_some_and(|t| matches!(t, Type::Any))
        || else_ty.is_some_and(|t| matches!(t, Type::Any));
    let needs_any_upgrade = any_branch && !matches!(result_ty, Type::Any | Type::Nil | Type::Never);
    let merge_ty = if needs_any_upgrade {
        &Type::Any
    } else {
        result_ty
    };

    // Does this if produce a value?
    let has_value = cl_type(merge_ty).is_some() && (then_expr.is_some() || else_expr.is_some());
    let result_cl_type = cl_type_or_i64(merge_ty);
    if has_value {
        ctx.builder.append_block_param(merge_block, result_cl_type);
    }

    ctx.builder
        .ins()
        .brif(cond_bool, then_block, &[], else_block, &[]);

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
        ctx.builder
            .ins()
            .trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
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
        ctx.builder
            .ins()
            .trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
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

/// Check if a ForRange loop body is safe to unroll.
/// Criteria: no break/continue, no function calls, no nested loops, no returns.
fn can_unroll_loop(body: &[HirStmt]) -> bool {
    for stmt in body {
        if !stmt_safe_to_unroll(stmt) {
            return false;
        }
    }
    true
}

fn stmt_safe_to_unroll(stmt: &HirStmt) -> bool {
    match stmt {
        HirStmt::Break | HirStmt::Continue | HirStmt::Return(_) => false,
        HirStmt::Import(_) => false,
        HirStmt::Assign { value, .. } => expr_safe_to_unroll(value),
        HirStmt::IndexAssign {
            target,
            index,
            value,
        } => {
            expr_safe_to_unroll(target) && expr_safe_to_unroll(index) && expr_safe_to_unroll(value)
        }
        HirStmt::MemberAssign { target, value, .. } => {
            expr_safe_to_unroll(target) && expr_safe_to_unroll(value)
        }
        HirStmt::Expr(e) => expr_safe_to_unroll(e),
        HirStmt::FuncDecl { .. } => false,
    }
}

fn expr_safe_to_unroll(expr: &HirExpr) -> bool {
    use HirExprKind::*;
    match &expr.kind {
        Int(_) | Float(_) | Str(_) | Bool(_) | Nil | Ident(_) => true,
        BinOp { left, right, .. } => expr_safe_to_unroll(left) && expr_safe_to_unroll(right),
        UnaryOp { operand, .. } => expr_safe_to_unroll(operand),
        // No function calls, no loops, no complex expressions
        Call { .. } | RuntimeCall { .. } | Lambda { .. } | Loop { .. } => false,
        If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            expr_safe_to_unroll(cond)
                && then_body.iter().all(stmt_safe_to_unroll)
                && then_expr.as_ref().is_none_or(|e| expr_safe_to_unroll(e))
                && else_body.iter().all(stmt_safe_to_unroll)
                && else_expr.as_ref().is_none_or(|e| expr_safe_to_unroll(e))
        }
        Index { target, index } => expr_safe_to_unroll(target) && expr_safe_to_unroll(index),
        Member { target, .. } => expr_safe_to_unroll(target),
        Array(elems) => elems.iter().all(expr_safe_to_unroll),
        Map(entries) => entries.iter().all(|(_, e)| expr_safe_to_unroll(e)),
        Tuple(elems) => elems.iter().all(expr_safe_to_unroll),
        Block { stmts, expr } => {
            stmts.iter().all(stmt_safe_to_unroll)
                && expr.as_ref().is_none_or(|e| expr_safe_to_unroll(e))
        }
        Length(e) => expr_safe_to_unroll(e),
        Range { start, end, .. } => expr_safe_to_unroll(start) && expr_safe_to_unroll(end),
        Go(_) | Receive(_) | Send { .. } | Select(_) => false,
    }
}

const UNROLL_FACTOR: i64 = 4;

fn compile_loop(ctx: &mut FuncCtx, kind: &HirLoopKind, body: &[HirStmt]) {
    match kind {
        HirLoopKind::While(cond) => {
            let header_block = ctx.builder.create_block();
            let body_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            ctx.builder.ins().jump(header_block, &[]);
            ctx.builder.switch_to_block(header_block);

            let cond_val =
                compile_expr(ctx, cond).expect("codegen: condition expr produced no value");
            let cond_bool = to_bool(ctx, cond_val, &cond.ty);
            ctx.builder
                .ins()
                .brif(cond_bool, body_block, &[], exit_block, &[]);

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
            let start_val =
                compile_expr(ctx, start).expect("codegen: start expr produced no value");
            let end_val = compile_expr(ctx, end).expect("codegen: end expr produced no value");

            // Create loop variable
            let loop_var = ctx.new_var(types::I64);
            ctx.builder.def_var(loop_var, start_val);
            ctx.vars.insert(var.clone(), (loop_var, Type::Int));

            let cc_asc = if *inclusive {
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThanOrEqual
            } else {
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan
            };

            // Determine if range is statically ascending (for unrolling optimization)
            let static_ascending = match (&start.kind, &end.kind) {
                (HirExprKind::Int(s), HirExprKind::Int(e)) => *s < *e,
                _ => false,
            };

            // Try loop unrolling for simple bodies (ascending, non-inclusive ranges only)
            if !*inclusive && can_unroll_loop(body) && static_ascending {
                let cc = cc_asc;
                // Unrolled loop: main loop steps by UNROLL_FACTOR, remainder loop handles leftovers
                let unrolled_body_block = ctx.builder.create_block();
                let unrolled_inc_block = ctx.builder.create_block();
                let remainder_body_block = ctx.builder.create_block();
                let remainder_inc_block = ctx.builder.create_block();
                let exit_block = ctx.builder.create_block();

                // Compute unrolled_end = start + ((end - start) / UNROLL_FACTOR) * UNROLL_FACTOR
                // This is the limit for the main unrolled loop
                let range_size = ctx.builder.ins().isub(end_val, start_val);
                let factor = ctx.builder.ins().iconst(types::I64, UNROLL_FACTOR);
                let full_chunks = ctx.builder.ins().sdiv(range_size, factor);
                let unrolled_count = ctx.builder.ins().imul(full_chunks, factor);
                let unrolled_end = ctx.builder.ins().iadd(start_val, unrolled_count);

                // Guard: skip unrolled loop if start >= unrolled_end (fewer than UNROLL_FACTOR iterations)
                let guard1 = ctx.builder.ins().icmp(cc, start_val, unrolled_end);
                ctx.builder
                    .ins()
                    .brif(guard1, unrolled_body_block, &[], remainder_body_block, &[]);

                // === Unrolled main loop body ===
                ctx.builder.switch_to_block(unrolled_body_block);

                // Push loop stack for break/continue (though unrolled bodies shouldn't have them)
                ctx.loop_stack.push((unrolled_inc_block, exit_block));

                // Emit body UNROLL_FACTOR times with loop var offset
                for u in 0..UNROLL_FACTOR {
                    if u > 0 {
                        let current = ctx.builder.use_var(loop_var);
                        let offset = ctx.builder.ins().iconst(types::I64, 1);
                        let next = ctx.builder.ins().iadd(current, offset);
                        ctx.builder.def_var(loop_var, next);
                    }
                    compile_body(ctx, body, &Type::Nil);
                }

                ctx.loop_stack.pop();

                if !ctx.block_terminated {
                    ctx.builder.ins().jump(unrolled_inc_block, &[]);
                }

                // Unrolled increment: advance by 1 more (total UNROLL_FACTOR) and check
                ctx.builder.switch_to_block(unrolled_inc_block);
                ctx.builder.seal_block(unrolled_inc_block);
                let current = ctx.builder.use_var(loop_var);
                let one = ctx.builder.ins().iconst(types::I64, 1);
                let next = ctx.builder.ins().iadd(current, one);
                ctx.builder.def_var(loop_var, next);
                let cond = ctx.builder.ins().icmp(cc, next, unrolled_end);
                ctx.builder
                    .ins()
                    .brif(cond, unrolled_body_block, &[], remainder_body_block, &[]);

                ctx.builder.seal_block(unrolled_body_block);

                // === Remainder loop (handles leftover iterations) ===
                ctx.builder.switch_to_block(remainder_body_block);

                // Check if there are any remaining iterations
                let rem_current = ctx.builder.use_var(loop_var);
                let rem_guard = ctx.builder.ins().icmp(cc, rem_current, end_val);
                let remainder_real_block = ctx.builder.create_block();
                ctx.builder
                    .ins()
                    .brif(rem_guard, remainder_real_block, &[], exit_block, &[]);

                ctx.builder.seal_block(remainder_body_block);

                // Remainder body
                ctx.builder.switch_to_block(remainder_real_block);

                ctx.loop_stack.push((remainder_inc_block, exit_block));
                compile_body(ctx, body, &Type::Nil);
                ctx.loop_stack.pop();

                if !ctx.block_terminated {
                    ctx.builder.ins().jump(remainder_inc_block, &[]);
                }

                // Remainder increment
                ctx.builder.switch_to_block(remainder_inc_block);
                ctx.builder.seal_block(remainder_inc_block);
                let rem_cur = ctx.builder.use_var(loop_var);
                let rem_one = ctx.builder.ins().iconst(types::I64, 1);
                let rem_next = ctx.builder.ins().iadd(rem_cur, rem_one);
                ctx.builder.def_var(loop_var, rem_next);
                let rem_cond = ctx.builder.ins().icmp(cc, rem_next, end_val);
                ctx.builder
                    .ins()
                    .brif(rem_cond, remainder_real_block, &[], exit_block, &[]);

                ctx.builder.seal_block(remainder_real_block);
                ctx.builder.switch_to_block(exit_block);
                ctx.builder.seal_block(exit_block);
                ctx.block_terminated = false;
            } else {
                // Standard rotated loop with runtime direction support.
                // Handles both ascending (0..5) and descending (5..0) ranges.
                let cc_desc = if *inclusive {
                    cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThanOrEqual
                } else {
                    cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThan
                };

                let body_block = ctx.builder.create_block();
                let dispatch_block = ctx.builder.create_block();
                let asc_inc_block = ctx.builder.create_block();
                let desc_inc_block = ctx.builder.create_block();
                let exit_block = ctx.builder.create_block();

                // Determine direction at runtime
                let is_ascending = ctx.builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                    start_val,
                    end_val,
                );

                // Guard: check if range has any iterations in either direction
                let asc_guard = ctx.builder.ins().icmp(cc_asc, start_val, end_val);
                let desc_guard = ctx.builder.ins().icmp(cc_desc, start_val, end_val);
                // Enter loop if ascending guard OR descending guard passes
                let enter_loop = ctx.builder.ins().bor(asc_guard, desc_guard);
                ctx.builder
                    .ins()
                    .brif(enter_loop, body_block, &[], exit_block, &[]);

                // Body block
                ctx.builder.switch_to_block(body_block);

                // continue jumps to dispatch_block which routes to correct inc
                ctx.loop_stack.push((dispatch_block, exit_block));
                compile_body(ctx, body, &Type::Nil);
                ctx.loop_stack.pop();

                if !ctx.block_terminated {
                    ctx.builder.ins().jump(dispatch_block, &[]);
                }

                // Dispatch block: route to ascending or descending increment
                ctx.builder.switch_to_block(dispatch_block);
                ctx.builder.seal_block(dispatch_block);
                ctx.builder
                    .ins()
                    .brif(is_ascending, asc_inc_block, &[], desc_inc_block, &[]);

                // Ascending increment: i += 1, check i < end
                ctx.builder.switch_to_block(asc_inc_block);
                ctx.builder.seal_block(asc_inc_block);
                let current = ctx.builder.use_var(loop_var);
                let one = ctx.builder.ins().iconst(types::I64, 1);
                let next = ctx.builder.ins().iadd(current, one);
                ctx.builder.def_var(loop_var, next);
                let asc_cond = ctx.builder.ins().icmp(cc_asc, next, end_val);
                ctx.builder
                    .ins()
                    .brif(asc_cond, body_block, &[], exit_block, &[]);

                // Descending increment: i -= 1, check i > end
                ctx.builder.switch_to_block(desc_inc_block);
                ctx.builder.seal_block(desc_inc_block);
                let current = ctx.builder.use_var(loop_var);
                let one = ctx.builder.ins().iconst(types::I64, 1);
                let next = ctx.builder.ins().isub(current, one);
                ctx.builder.def_var(loop_var, next);
                let desc_cond = ctx.builder.ins().icmp(cc_desc, next, end_val);
                ctx.builder
                    .ins()
                    .brif(desc_cond, body_block, &[], exit_block, &[]);

                ctx.builder.seal_block(body_block);
                ctx.builder.switch_to_block(exit_block);
                ctx.builder.seal_block(exit_block);
                ctx.block_terminated = false;
            }
        }

        HirLoopKind::ForEach { var, iter } => {
            let iter_val =
                compile_expr(ctx, iter).expect("codegen: iterator expr produced no value");

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
            ctx.builder
                .ins()
                .brif(cond, body_block, &[], exit_block, &[]);

            ctx.builder.switch_to_block(body_block);
            ctx.builder.seal_block(body_block);

            // Get element
            let current_idx = ctx.builder.use_var(idx_var);
            if is_any_iter {
                // For Any, use tok_value_index which dispatches by tag
                let (t_tag, t_data) = to_tokvalue(ctx, iter_val, &iter.ty);
                let idx_ref = ctx.get_runtime_func_ref("tok_value_index");
                let idx_call = ctx
                    .builder
                    .ins()
                    .call(idx_ref, &[t_tag, t_data, current_idx]);
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
            // Indexed foreach: ~(i v:collection){...}
            // For arrays: i = integer index, v = element
            // For maps: i = string key, v = value
            let iter_val =
                compile_expr(ctx, iter).expect("codegen: iterator expr produced no value");
            let is_map = matches!(&iter.ty, Type::Map(_));

            // Get length and (for maps) extract keys/vals arrays
            let (len_val, keys_arr, vals_arr) = if is_map {
                let len_ref = ctx.get_runtime_func_ref("tok_map_len");
                let len_call = ctx.builder.ins().call(len_ref, &[iter_val]);
                let len_val = ctx.builder.inst_results(len_call)[0];

                let keys_ref = ctx.get_runtime_func_ref("tok_map_keys");
                let keys_call = ctx.builder.ins().call(keys_ref, &[iter_val]);
                let keys_arr = ctx.builder.inst_results(keys_call)[0];

                let vals_ref = ctx.get_runtime_func_ref("tok_map_vals");
                let vals_call = ctx.builder.ins().call(vals_ref, &[iter_val]);
                let vals_arr = ctx.builder.inst_results(vals_call)[0];

                (len_val, Some(keys_arr), Some(vals_arr))
            } else {
                let len_ref = ctx.get_runtime_func_ref("tok_array_len");
                let len_call = ctx.builder.ins().call(len_ref, &[iter_val]);
                let len_val = ctx.builder.inst_results(len_call)[0];
                (len_val, None, None)
            };

            // Internal integer loop counter (always i64)
            let int_idx_var = ctx.new_var(types::I64);
            let zero = ctx.builder.ins().iconst(types::I64, 0);
            ctx.builder.def_var(int_idx_var, zero);

            // User-visible index/key variable
            let idx_type = if is_map { Type::Str } else { Type::Int };
            let idx_ct = cl_type_or_i64(&idx_type);
            let idx_var = ctx.new_var(idx_ct);
            let idx_zero = zero_value(&mut ctx.builder, idx_ct);
            ctx.builder.def_var(idx_var, idx_zero);
            ctx.vars.insert(idx_name.clone(), (idx_var, idx_type));

            // Value variable
            let elem_type = match &iter.ty {
                Type::Array(inner) => inner.as_ref().clone(),
                Type::Map(inner) => inner.as_ref().clone(),
                _ => Type::Any,
            };
            let ct = cl_type_or_i64(&elem_type);
            let elem_var = ctx.new_var(ct);
            let elem_zero = zero_value(&mut ctx.builder, ct);
            ctx.builder.def_var(elem_var, elem_zero);
            ctx.vars
                .insert(val_name.clone(), (elem_var, elem_type.clone()));

            let header_block = ctx.builder.create_block();
            let body_block = ctx.builder.create_block();
            let inc_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            ctx.builder.ins().jump(header_block, &[]);
            ctx.builder.switch_to_block(header_block);

            let current_idx = ctx.builder.use_var(int_idx_var);
            let cond = ctx.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                current_idx,
                len_val,
            );
            ctx.builder
                .ins()
                .brif(cond, body_block, &[], exit_block, &[]);

            ctx.builder.switch_to_block(body_block);
            ctx.builder.seal_block(body_block);

            let current_idx = ctx.builder.use_var(int_idx_var);

            if is_map {
                // Map iteration: fetch key from keys array, value from vals array
                let get_ref = ctx.get_runtime_func_ref("tok_array_get");

                let get_key_call = ctx.builder.ins().call(
                    get_ref,
                    &[
                        keys_arr.expect("codegen: map iteration missing keys array"),
                        current_idx,
                    ],
                );
                let key_results = ctx.builder.inst_results(get_key_call);
                let key = from_tokvalue(ctx, key_results[0], key_results[1], &Type::Str);
                ctx.builder.def_var(idx_var, key);

                let get_val_call = ctx.builder.ins().call(
                    get_ref,
                    &[
                        vals_arr.expect("codegen: map iteration missing vals array"),
                        current_idx,
                    ],
                );
                let val_results = ctx.builder.inst_results(get_val_call);
                let val = from_tokvalue(ctx, val_results[0], val_results[1], &elem_type);
                ctx.builder.def_var(elem_var, val);
            } else {
                // Array iteration: index is the integer, value from array
                ctx.builder.def_var(idx_var, current_idx);

                let get_ref = ctx.get_runtime_func_ref("tok_array_get");
                let get_call = ctx.builder.ins().call(get_ref, &[iter_val, current_idx]);
                let results = ctx.builder.inst_results(get_call);
                let elem = from_tokvalue(ctx, results[0], results[1], &elem_type);
                ctx.builder.def_var(elem_var, elem);
            }

            ctx.loop_stack.push((inc_block, exit_block));
            compile_body(ctx, body, &Type::Nil);
            ctx.loop_stack.pop();

            if !ctx.block_terminated {
                ctx.builder.ins().jump(inc_block, &[]);
            }

            ctx.builder.switch_to_block(inc_block);
            ctx.builder.seal_block(inc_block);
            let current_idx = ctx.builder.use_var(int_idx_var);
            let one = ctx.builder.ins().iconst(types::I64, 1);
            let next_idx = ctx.builder.ins().iadd(current_idx, one);
            ctx.builder.def_var(int_idx_var, next_idx);
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
        ctx.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), val, 8)
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
            let tag = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), val, 0);
            let data = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), val, 8);
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
        Type::Str
        | Type::Array(_)
        | Type::Map(_)
        | Type::Tuple(_)
        | Type::Func(_)
        | Type::Channel(_)
        | Type::Handle(_) => data, // pointer
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
            ctx.builder
                .ins()
                .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, val, zero)
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
            ctx.builder
                .ins()
                .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, val, zero)
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
    stmts
        .into_iter()
        .map(|s| match s {
            HirStmt::Return(Some(expr)) => HirStmt::Expr(expr),
            HirStmt::Return(None) => HirStmt::Expr(HirExpr {
                kind: HirExprKind::Nil,
                ty: Type::Nil,
            }),
            other => other,
        })
        .collect()
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
                ty: if matches!(ty, Type::Any) {
                    new_value.ty.clone()
                } else {
                    ty.clone()
                },
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
            **left = retype_expr(left, type_map);
            **right = retype_expr(right, type_map);
            // Propagate: infer result type from children
            e.ty = infer_binop_type(&left.ty, &right.ty, *op);
        }
        HirExprKind::UnaryOp { operand, op: _ } => {
            **operand = retype_expr(operand, type_map);
            // Neg preserves type, Not → Bool
            e.ty = match &operand.ty {
                Type::Int => Type::Int,
                Type::Float => Type::Float,
                _ => e.ty.clone(),
            };
        }
        HirExprKind::Call { func, args } => {
            **func = retype_expr(func, type_map);
            for arg in args.iter_mut() {
                *arg = retype_expr(arg, type_map);
            }
            // Don't change the call's result type — it depends on the callee
        }
        // Literals keep their types — no rewriting needed
        HirExprKind::Int(_)
        | HirExprKind::Float(_)
        | HirExprKind::Bool(_)
        | HirExprKind::Str(_)
        | HirExprKind::Nil => {}
        // For other complex nodes, just leave as-is
        _ => {}
    }
    e
}

/// Infer the result type of a binary operation given the types of both operands.
/// Delegates to the canonical `tok_types::infer_binop_type` via `HirBinOp::to_parser_op()`.
fn infer_binop_type(left: &Type, right: &Type, op: HirBinOp) -> Type {
    tok_types::infer_binop_type(&op.to_parser_op(), left, right)
}

// ─── Free variable analysis for closure captures ──────────────────────

/// Collect all free variables referenced in a lambda body that are not in `bound` (params + locals).
/// Returns the names of variables that need to be captured from the enclosing scope.
/// Maximum HIR nesting depth before aborting free-var collection.
/// Prevents stack overflow on adversarially nested input.
const MAX_FREE_VAR_DEPTH: usize = 1000;

fn collect_free_vars(body: &[HirStmt], param_names: &HashSet<String>) -> HashSet<String> {
    let mut free = HashSet::new();
    let mut locals = param_names.clone();
    for stmt in body {
        collect_free_vars_stmt(stmt, &mut locals, &mut free, 0);
    }
    free
}

fn collect_free_vars_stmt(
    stmt: &HirStmt,
    locals: &mut HashSet<String>,
    free: &mut HashSet<String>,
    depth: usize,
) {
    if depth >= MAX_FREE_VAR_DEPTH {
        return;
    }
    match stmt {
        HirStmt::Assign { name, value, .. } => {
            // The RHS may reference free vars (before the local is defined)
            collect_free_vars_expr(value, locals, free, depth + 1);
            locals.insert(name.clone());
        }
        HirStmt::Expr(expr) => {
            collect_free_vars_expr(expr, locals, free, depth + 1);
        }
        HirStmt::Return(opt_expr) => {
            if let Some(expr) = opt_expr {
                collect_free_vars_expr(expr, locals, free, depth + 1);
            }
        }
        HirStmt::FuncDecl { name, .. } => {
            // The function name becomes a local, its body is a separate scope
            locals.insert(name.clone());
        }
        HirStmt::IndexAssign {
            target,
            index,
            value,
            ..
        } => {
            collect_free_vars_expr(target, locals, free, depth + 1);
            collect_free_vars_expr(index, locals, free, depth + 1);
            collect_free_vars_expr(value, locals, free, depth + 1);
        }
        HirStmt::MemberAssign { target, value, .. } => {
            collect_free_vars_expr(target, locals, free, depth + 1);
            collect_free_vars_expr(value, locals, free, depth + 1);
        }
        HirStmt::Import(_) => {}
        HirStmt::Break | HirStmt::Continue => {}
    }
}

fn collect_free_vars_expr(
    expr: &HirExpr,
    locals: &HashSet<String>,
    free: &mut HashSet<String>,
    depth: usize,
) {
    if depth >= MAX_FREE_VAR_DEPTH {
        return;
    }
    let d = depth + 1;
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
                collect_free_vars_expr(e, locals, free, d);
            }
        }
        HirExprKind::Map(entries) => {
            for (_k, v) in entries {
                collect_free_vars_expr(v, locals, free, d);
            }
        }
        HirExprKind::Tuple(elems) => {
            for e in elems {
                collect_free_vars_expr(e, locals, free, d);
            }
        }
        HirExprKind::BinOp { left, right, .. } => {
            collect_free_vars_expr(left, locals, free, d);
            collect_free_vars_expr(right, locals, free, d);
        }
        HirExprKind::UnaryOp { operand, .. } => {
            collect_free_vars_expr(operand, locals, free, d);
        }
        HirExprKind::Index { target, index } => {
            collect_free_vars_expr(target, locals, free, d);
            collect_free_vars_expr(index, locals, free, d);
        }
        HirExprKind::Member { target, .. } => {
            collect_free_vars_expr(target, locals, free, d);
        }
        HirExprKind::Call { func, args } => {
            collect_free_vars_expr(func, locals, free, d);
            for a in args {
                collect_free_vars_expr(a, locals, free, d);
            }
        }
        HirExprKind::RuntimeCall { args, .. } => {
            for a in args {
                collect_free_vars_expr(a, locals, free, d);
            }
        }
        HirExprKind::If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            collect_free_vars_expr(cond, locals, free, d);
            for s in then_body {
                collect_free_vars_stmt(s, &mut locals.clone(), free, d);
            }
            if let Some(e) = then_expr {
                collect_free_vars_expr(e, locals, free, d);
            }
            for s in else_body {
                collect_free_vars_stmt(s, &mut locals.clone(), free, d);
            }
            if let Some(e) = else_expr {
                collect_free_vars_expr(e, locals, free, d);
            }
        }
        HirExprKind::Loop { kind, body } => {
            // Collect free vars from the loop kind (range start/end, condition, iterator)
            let mut loop_locals = locals.clone();
            match kind.as_ref() {
                HirLoopKind::ForRange {
                    var, start, end, ..
                } => {
                    collect_free_vars_expr(start, locals, free, d);
                    collect_free_vars_expr(end, locals, free, d);
                    loop_locals.insert(var.clone());
                }
                HirLoopKind::ForEach { var, iter } => {
                    collect_free_vars_expr(iter, locals, free, d);
                    loop_locals.insert(var.clone());
                }
                HirLoopKind::ForEachIndexed {
                    idx_var,
                    val_var,
                    iter,
                } => {
                    collect_free_vars_expr(iter, locals, free, d);
                    loop_locals.insert(idx_var.clone());
                    loop_locals.insert(val_var.clone());
                }
                HirLoopKind::While(cond) => {
                    collect_free_vars_expr(cond, locals, free, d);
                }
                HirLoopKind::Infinite => {}
            }
            for s in body {
                collect_free_vars_stmt(s, &mut loop_locals, free, d);
            }
        }
        HirExprKind::Lambda { params, body, .. } => {
            // Nested lambda: its params are bound, but it may capture from our scope
            let mut inner_locals = locals.clone();
            for p in params {
                inner_locals.insert(p.name.clone());
            }
            for s in body {
                collect_free_vars_stmt(s, &mut inner_locals, free, d);
            }
        }
        HirExprKind::Length(inner) => {
            collect_free_vars_expr(inner, locals, free, d);
        }
        HirExprKind::Block { stmts, expr } => {
            let mut block_locals = locals.clone();
            for s in stmts {
                collect_free_vars_stmt(s, &mut block_locals, free, d);
            }
            if let Some(e) = expr {
                collect_free_vars_expr(e, &block_locals, free, d);
            }
        }
        HirExprKind::Range { start, end, .. } => {
            collect_free_vars_expr(start, locals, free, d);
            collect_free_vars_expr(end, locals, free, d);
        }
        HirExprKind::Go(inner) => {
            collect_free_vars_expr(inner, locals, free, d);
        }
        HirExprKind::Receive(inner) => {
            collect_free_vars_expr(inner, locals, free, d);
        }
        HirExprKind::Send { chan, value } => {
            collect_free_vars_expr(chan, locals, free, d);
            collect_free_vars_expr(value, locals, free, d);
        }
        HirExprKind::Select(arms) => {
            for arm in arms {
                match arm {
                    HirSelectArm::Recv { chan, body, .. } => {
                        collect_free_vars_expr(chan, locals, free, d);
                        for s in body {
                            collect_free_vars_stmt(s, &mut locals.clone(), free, d);
                        }
                    }
                    HirSelectArm::Send { chan, value, body } => {
                        collect_free_vars_expr(chan, locals, free, d);
                        collect_free_vars_expr(value, locals, free, d);
                        for s in body {
                            collect_free_vars_stmt(s, &mut locals.clone(), free, d);
                        }
                    }
                    HirSelectArm::Default(body) => {
                        for s in body {
                            collect_free_vars_stmt(s, &mut locals.clone(), free, d);
                        }
                    }
                }
            }
        }
    }
}
