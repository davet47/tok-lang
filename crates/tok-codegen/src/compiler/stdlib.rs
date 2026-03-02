// ─── Stdlib module registry (single source of truth) ──────────────────

/// Mapping from stdlib module name to its runtime constructor symbol.
///
/// This is the authoritative list of all stdlib modules. Both the codegen
/// import handler and the driver's `is_stdlib_module()` check use this table.
pub(crate) const STDLIB_MODULE_CONSTRUCTORS: &[(&str, &str)] = &[
    ("math", "tok_stdlib_math"),
    ("str", "tok_stdlib_str"),
    ("os", "tok_stdlib_os"),
    ("io", "tok_stdlib_io"),
    ("json", "tok_stdlib_json"),
    ("llm", "tok_stdlib_llm"),
    ("csv", "tok_stdlib_csv"),
    ("fs", "tok_stdlib_fs"),
    ("http", "tok_stdlib_http"),
    ("re", "tok_stdlib_re"),
    ("time", "tok_stdlib_time"),
    ("tmpl", "tok_stdlib_tmpl"),
    ("toon", "tok_stdlib_toon"),
];

/// Check if a name is a known stdlib module (not a file import).
pub fn is_stdlib_module(name: &str) -> bool {
    STDLIB_MODULE_CONSTRUCTORS.iter().any(|(n, _)| *n == name)
}

/// Get the runtime constructor symbol for a stdlib module name.
pub(crate) fn stdlib_constructor(name: &str) -> Option<&'static str> {
    STDLIB_MODULE_CONSTRUCTORS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, sym)| *sym)
}

/// Look up a stdlib direct-call function: returns (trampoline_symbol, arity).
pub(crate) fn get_stdlib_func(module: &str, field: &str) -> Option<(&'static str, usize)> {
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
pub(crate) fn get_stdlib_const(module: &str, field: &str) -> Option<f64> {
    match (module, field) {
        ("math", "pi") => Some(std::f64::consts::PI),
        ("math", "e") => Some(std::f64::consts::E),
        ("math", "inf") => Some(f64::INFINITY),
        ("math", "nan") => Some(f64::NAN),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_module::Module;

    /// All (module, field) pairs registered in get_stdlib_func.
    /// This must be kept in sync with the match arms above.
    const ALL_STDLIB_FUNCS: &[(&str, &str)] = &[
        // math
        ("math", "sqrt"),
        ("math", "sin"),
        ("math", "cos"),
        ("math", "tan"),
        ("math", "asin"),
        ("math", "acos"),
        ("math", "atan"),
        ("math", "log"),
        ("math", "log2"),
        ("math", "log10"),
        ("math", "exp"),
        ("math", "floor"),
        ("math", "ceil"),
        ("math", "round"),
        ("math", "abs"),
        ("math", "pow"),
        ("math", "min"),
        ("math", "max"),
        ("math", "atan2"),
        ("math", "random"),
        // str
        ("str", "upper"),
        ("str", "lower"),
        ("str", "trim"),
        ("str", "trim_left"),
        ("str", "trim_right"),
        ("str", "chars"),
        ("str", "bytes"),
        ("str", "rev"),
        ("str", "len"),
        ("str", "contains"),
        ("str", "starts_with"),
        ("str", "ends_with"),
        ("str", "index_of"),
        ("str", "repeat"),
        ("str", "split"),
        ("str", "replace"),
        ("str", "pad_left"),
        ("str", "pad_right"),
        ("str", "substr"),
        // json
        ("json", "jparse"),
        ("json", "jstr"),
        ("json", "jpretty"),
        ("json", "parse"),
        ("json", "stringify"),
        ("json", "pretty"),
        // llm
        ("llm", "ask"),
        ("llm", "chat"),
        // csv
        ("csv", "cparse"),
        ("csv", "cstr"),
        ("csv", "parse"),
        ("csv", "stringify"),
        // tmpl
        ("tmpl", "render"),
        ("tmpl", "apply"),
        ("tmpl", "compile"),
        // toon
        ("toon", "tparse"),
        ("toon", "tstr"),
        ("toon", "parse"),
        ("toon", "stringify"),
        // os
        ("os", "args"),
        ("os", "cwd"),
        ("os", "pid"),
        ("os", "env"),
        ("os", "exit"),
        ("os", "exec"),
        ("os", "set_env"),
        // io
        ("io", "readall"),
        ("io", "input"),
        // fs
        ("fs", "fread"),
        ("fs", "fexists"),
        ("fs", "fls"),
        ("fs", "fmk"),
        ("fs", "frm"),
        ("fs", "fwrite"),
        ("fs", "fappend"),
        // http
        ("http", "hget"),
        ("http", "hdel"),
        ("http", "hpost"),
        ("http", "hput"),
        ("http", "serve"),
        // re
        ("re", "rmatch"),
        ("re", "rfind"),
        ("re", "rall"),
        ("re", "rsub"),
        // time
        ("time", "now"),
        ("time", "sleep"),
        ("time", "fmt"),
    ];

    #[test]
    fn all_stdlib_funcs_resolve() {
        for &(module, field) in ALL_STDLIB_FUNCS {
            assert!(
                get_stdlib_func(module, field).is_some(),
                "get_stdlib_func({module:?}, {field:?}) returned None"
            );
        }
    }

    #[test]
    fn stdlib_arity_matches_runtime_decl() {
        // Verify that each stdlib function's arity maps to the correct
        // number of Cranelift params in the runtime declaration.
        // Trampoline convention: (env: PTR, tag1: I64, data1: I64, ...) -> (I64, I64)
        // So param count = 1 (env) + 2 * arity (tag+data per arg)
        let mut compiler = crate::compiler::Compiler::new();
        compiler.declare_all_runtime_funcs();

        for &(module, field) in ALL_STDLIB_FUNCS {
            let (trampoline, arity) = get_stdlib_func(module, field)
                .unwrap_or_else(|| panic!("missing: {module}.{field}"));

            let func_id = compiler
                .runtime_funcs
                .get(trampoline)
                .unwrap_or_else(|| panic!("trampoline {trampoline} not declared in runtime_decls"));

            let decl = compiler.module.declarations().get_function_decl(*func_id);
            let expected_params = 1 + 2 * arity; // env + (tag, data) per arg
            assert_eq!(
                decl.signature.params.len(),
                expected_params,
                "{module}.{field} ({trampoline}): arity={arity}, \
                 expected {expected_params} params, got {}",
                decl.signature.params.len()
            );
        }
    }

    #[test]
    fn all_modules_have_constructor() {
        let modules: Vec<&str> = ALL_STDLIB_FUNCS
            .iter()
            .map(|(m, _)| *m)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        for module in modules {
            assert!(
                stdlib_constructor(module).is_some(),
                "module {module:?} has functions but no constructor in STDLIB_MODULE_CONSTRUCTORS"
            );
        }
    }
}
