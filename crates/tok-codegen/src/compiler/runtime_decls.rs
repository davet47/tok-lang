// ─── Runtime function declarations ────────────────────────────────────

use cranelift_codegen::ir::types;

use super::{Compiler, PTR};

impl Compiler {
    /// Declare all runtime functions we'll need.
    pub(crate) fn declare_all_runtime_funcs(&mut self) {
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
        self.declare_runtime_func("tok_map_clone", &[PTR], &[PTR]);

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
        self.declare_runtime_func(
            "tok_value_index_set",
            &[PTR, types::I64, PTR, types::I64, PTR, types::I64],
            &[],
        );

        // Closure
        self.declare_runtime_func(
            "tok_closure_alloc",
            &[PTR, PTR, types::I32, types::I32],
            &[PTR],
        );
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
        self.declare_runtime_func("tok_value_rc_inc", &[types::I64, types::I64], &[]);
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
}
