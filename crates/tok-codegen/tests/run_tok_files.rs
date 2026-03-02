/// Integration tests: compile .tok files to native executables, run them,
/// and validate output against `// expect:` annotations.
///
/// This is the most comprehensive test tier — it validates the entire pipeline
/// from source to correct runtime behavior.
mod common;

use std::fs;
use std::path::Path;

/// Files to skip entirely for compile+run tests (not in run_dir catch-all either).
const RUN_SKIP: &[&str] = &[
    // Interactive / non-deterministic
    "stdlib_io_test.tok",    // reads stdin
    "stdlib_http_test.tok",  // makes network requests
    "stdlib_llm_test.tok",   // needs API key
    "stdlib_time_test.tok",  // timing-dependent output
    "stdlib_os_test.tok",    // exec/env/pid varies by system
    "builtins_new_test.tok", // clock/env/args vary by system
    // Needs import resolver (tok-driver)
    "imports_test.tok",
    // Known codegen bugs
    "stdlib_tmpl_test.tok", // argument count mismatch
    // Pre-existing runtime crashes (codegen limitations)
    "errors_tuples_test.tok",      // runtime crash
    "concurrency_test.tok",        // runtime crash
    "stdlib_destructure_test.tok", // runtime crash
    // Not standalone programs
    "math_helpers.tok", // helper module for imports
];

fn run_file(path: &Path) {
    common::compile_run_and_validate_file(path);
}

fn run_dir(dir: &Path) {
    if !dir.exists() {
        return;
    }
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "tok"))
        .collect();
    files.sort();

    for path in files {
        let filename = path.file_name().unwrap().to_str().unwrap();
        if filename.starts_with('_')
            || filename.starts_with("debug")
            || common::should_skip_run(filename)
            || RUN_SKIP.contains(&filename)
        {
            continue;
        }
        run_file(&path);
    }
}

// ─── Core language tests (passing) ────────────────────────────────

#[test]
fn run_basics_test() {
    run_file(&common::tests_dir().join("basics_test.tok"));
}

#[test]
fn run_basics_compile() {
    run_file(&common::tests_dir().join("basics_compile.tok"));
}

#[test]
fn run_control_flow_test() {
    run_file(&common::tests_dir().join("control_flow_test.tok"));
}

#[test]
fn run_functions_test() {
    run_file(&common::tests_dir().join("functions_test.tok"));
}

#[test]
fn run_arrays_compile() {
    run_file(&common::tests_dir().join("arrays_compile.tok"));
}

#[test]
fn run_strings_compile() {
    run_file(&common::tests_dir().join("strings_compile.tok"));
}

#[test]
fn run_maps_compile() {
    run_file(&common::tests_dir().join("maps_compile.tok"));
}

#[test]
fn run_coverage_test() {
    run_file(&common::tests_dir().join("coverage_test.tok"));
}

#[test]
fn run_hello() {
    run_file(&common::tests_dir().join("hello.tok"));
}

#[test]
fn run_simple_func() {
    run_file(&common::tests_dir().join("simple_func.tok"));
}

#[test]
fn run_any_test() {
    run_file(&common::tests_dir().join("any_test.tok"));
}

// ─── Feature tests (passing) ──────────────────────────────────────

#[test]
fn run_default_params_test() {
    run_file(&common::tests_dir().join("default_params_test.tok"));
}

#[test]
fn run_variadic_test() {
    run_file(&common::tests_dir().join("variadic_test.tok"));
}

#[test]
fn run_spread_args_test() {
    run_file(&common::tests_dir().join("spread_args_test.tok"));
}

#[test]
fn run_prototype_test() {
    run_file(&common::tests_dir().join("prototype_test.tok"));
}

// ─── Stdlib tests (passing) ───────────────────────────────────────

#[test]
fn run_stdlib_math_test() {
    run_file(&common::tests_dir().join("stdlib_math_test.tok"));
}

#[test]
fn run_stdlib_str_test() {
    run_file(&common::tests_dir().join("stdlib_str_test.tok"));
}

#[test]
fn run_stdlib_json_test() {
    run_file(&common::tests_dir().join("stdlib_json_test.tok"));
}

#[test]
fn run_stdlib_csv_test() {
    run_file(&common::tests_dir().join("stdlib_csv_test.tok"));
}

#[test]
fn run_stdlib_fs_test() {
    run_file(&common::tests_dir().join("stdlib_fs_test.tok"));
}

#[test]
fn run_stdlib_re_test() {
    run_file(&common::tests_dir().join("stdlib_re_test.tok"));
}

#[test]
fn run_stdlib_toon_test() {
    run_file(&common::tests_dir().join("stdlib_toon_test.tok"));
}

// ─── Codegen regression tests ──────────────────────────────────────

#[test]
fn run_codegen_bool_values() {
    run_file(&common::tests_dir().join("codegen/bool_values.tok"));
}

#[test]
fn run_codegen_bool_short_circuit() {
    run_file(&common::tests_dir().join("codegen/bool_short_circuit.tok"));
}

#[test]
fn run_codegen_any_coercion() {
    run_file(&common::tests_dir().join("codegen/any_coercion.tok"));
}

#[test]
fn run_codegen_nested_ternary() {
    run_file(&common::tests_dir().join("codegen/nested_ternary.tok"));
}

#[test]
fn run_codegen_func_as_value() {
    run_file(&common::tests_dir().join("codegen/func_as_value.tok"));
}

#[test]
fn run_codegen_string_multiply() {
    run_file(&common::tests_dir().join("codegen/string_multiply.tok"));
}

#[test]
fn run_codegen_stdlib_direct() {
    run_file(&common::tests_dir().join("codegen/stdlib_direct.tok"));
}

#[test]
fn run_codegen_closure_capture() {
    run_file(&common::tests_dir().join("codegen/closure_capture.tok"));
}

#[test]
fn run_codegen_loop_collect() {
    run_file(&common::tests_dir().join("codegen/loop_collect.tok"));
}

#[test]
fn run_codegen_rc_basic() {
    run_file(&common::tests_dir().join("codegen/rc_basic.tok"));
}

#[test]
fn run_codegen_map_basic() {
    run_file(&common::tests_dir().join("codegen/map_basic.tok"));
}

// ─── Tests with known runtime crashes (#[ignore]) ──────────────────
// These tests correctly compile but crash at runtime due to pre-existing
// codegen limitations. They are kept as regression markers — when fixed,
// remove the #[ignore] to include them in the test suite.

#[test]
fn run_arrays_lambdas_test() {
    run_file(&common::tests_dir().join("arrays_lambdas_test.tok"));
}

#[test]
fn run_strings_pipes_test() {
    run_file(&common::tests_dir().join("strings_pipes_test.tok"));
}

#[test]
fn run_maps_test() {
    run_file(&common::tests_dir().join("maps_test.tok"));
}

#[test]
#[ignore] // Pre-existing: runtime crash (error handling/tuples)
fn run_errors_tuples_test() {
    run_file(&common::tests_dir().join("errors_tuples_test.tok"));
}

#[test]
#[ignore] // Pre-existing: runtime crash (goroutines/channels)
fn run_concurrency_test() {
    run_file(&common::tests_dir().join("concurrency_test.tok"));
}

#[test]
fn run_head_tail_test() {
    run_file(&common::tests_dir().join("head_tail_test.tok"));
}

#[test]
#[ignore] // Pre-existing: runtime crash (destructured imports)
fn run_stdlib_destructure_test() {
    run_file(&common::tests_dir().join("stdlib_destructure_test.tok"));
}

// ─── Catch-all ─────────────────────────────────────────────────────

#[test]
fn run_all_test_tok_files() {
    run_dir(&common::tests_dir());
}
