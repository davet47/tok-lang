/// Integration tests: lex → parse → typecheck → lower → **codegen** for all .tok test files.
///
/// Ensures the Cranelift codegen does not panic on any real Tok source file.
/// This extends the existing tok-hir lowering tests by adding the codegen pass.
mod common;

use std::fs;
use std::path::Path;

fn compile_file(path: &Path) {
    let source = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    let _obj_bytes = common::compile_source(&source);
    // If we got here without panicking, codegen succeeded.
}

/// Files that are known to fail codegen (need import resolver or have known bugs).
const COMPILE_SKIP: &[&str] = &[
    "imports_test.tok",      // needs import resolver (tok-driver)
    "stdlib_tmpl_test.tok",  // known codegen bug: argument count mismatch
];

fn compile_dir(dir: &Path) {
    if !dir.exists() {
        return;
    }
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "tok") {
            let filename = path.file_name().unwrap().to_str().unwrap();
            // Skip private modules (not standalone) and known failures
            if filename.starts_with('_') || COMPILE_SKIP.contains(&filename) {
                continue;
            }
            compile_file(&path);
        }
    }
}

// ─── Individual test files ─────────────────────────────────────────

#[test]
fn compile_basics_test() {
    compile_file(&common::tests_dir().join("basics_test.tok"));
}

#[test]
fn compile_basics_compile() {
    compile_file(&common::tests_dir().join("basics_compile.tok"));
}

#[test]
fn compile_control_flow_test() {
    compile_file(&common::tests_dir().join("control_flow_test.tok"));
}

#[test]
fn compile_functions_test() {
    compile_file(&common::tests_dir().join("functions_test.tok"));
}

#[test]
fn compile_arrays_lambdas_test() {
    compile_file(&common::tests_dir().join("arrays_lambdas_test.tok"));
}

#[test]
fn compile_strings_pipes_test() {
    compile_file(&common::tests_dir().join("strings_pipes_test.tok"));
}

#[test]
fn compile_maps_test() {
    compile_file(&common::tests_dir().join("maps_test.tok"));
}

#[test]
fn compile_errors_tuples_test() {
    compile_file(&common::tests_dir().join("errors_tuples_test.tok"));
}

#[test]
fn compile_concurrency_test() {
    compile_file(&common::tests_dir().join("concurrency_test.tok"));
}

#[test]
fn compile_coverage_test() {
    compile_file(&common::tests_dir().join("coverage_test.tok"));
}

#[test]
fn compile_arrays_compile() {
    compile_file(&common::tests_dir().join("arrays_compile.tok"));
}

#[test]
fn compile_strings_compile() {
    compile_file(&common::tests_dir().join("strings_compile.tok"));
}

#[test]
fn compile_maps_compile() {
    compile_file(&common::tests_dir().join("maps_compile.tok"));
}

#[test]
#[ignore] // Needs import resolver (lives in tok-driver, not available to tok-codegen)
fn compile_imports_test() {
    compile_file(&common::tests_dir().join("imports_test.tok"));
}

#[test]
fn compile_default_params_test() {
    compile_file(&common::tests_dir().join("default_params_test.tok"));
}

#[test]
fn compile_variadic_test() {
    compile_file(&common::tests_dir().join("variadic_test.tok"));
}

#[test]
fn compile_spread_args_test() {
    compile_file(&common::tests_dir().join("spread_args_test.tok"));
}

#[test]
fn compile_head_tail_test() {
    compile_file(&common::tests_dir().join("head_tail_test.tok"));
}

#[test]
fn compile_prototype_test() {
    compile_file(&common::tests_dir().join("prototype_test.tok"));
}

#[test]
fn compile_any_test() {
    compile_file(&common::tests_dir().join("any_test.tok"));
}

#[test]
fn compile_builtins_new_test() {
    compile_file(&common::tests_dir().join("builtins_new_test.tok"));
}

// ─── Stdlib tests ──────────────────────────────────────────────────

#[test]
fn compile_stdlib_math_test() {
    compile_file(&common::tests_dir().join("stdlib_math_test.tok"));
}

#[test]
fn compile_stdlib_str_test() {
    compile_file(&common::tests_dir().join("stdlib_str_test.tok"));
}

#[test]
fn compile_stdlib_json_test() {
    compile_file(&common::tests_dir().join("stdlib_json_test.tok"));
}

#[test]
fn compile_stdlib_csv_test() {
    compile_file(&common::tests_dir().join("stdlib_csv_test.tok"));
}

#[test]
fn compile_stdlib_os_test() {
    compile_file(&common::tests_dir().join("stdlib_os_test.tok"));
}

#[test]
fn compile_stdlib_io_test() {
    compile_file(&common::tests_dir().join("stdlib_io_test.tok"));
}

#[test]
fn compile_stdlib_fs_test() {
    compile_file(&common::tests_dir().join("stdlib_fs_test.tok"));
}

#[test]
fn compile_stdlib_re_test() {
    compile_file(&common::tests_dir().join("stdlib_re_test.tok"));
}

#[test]
fn compile_stdlib_time_test() {
    compile_file(&common::tests_dir().join("stdlib_time_test.tok"));
}

#[test]
fn compile_stdlib_toon_test() {
    compile_file(&common::tests_dir().join("stdlib_toon_test.tok"));
}

#[test]
#[ignore] // Known codegen bug: tmpl.apply/render argument count mismatch
fn compile_stdlib_tmpl_test() {
    compile_file(&common::tests_dir().join("stdlib_tmpl_test.tok"));
}

#[test]
fn compile_stdlib_http_test() {
    compile_file(&common::tests_dir().join("stdlib_http_test.tok"));
}

#[test]
fn compile_stdlib_llm_test() {
    compile_file(&common::tests_dir().join("stdlib_llm_test.tok"));
}

#[test]
fn compile_stdlib_destructure_test() {
    compile_file(&common::tests_dir().join("stdlib_destructure_test.tok"));
}

// ─── Codegen regression tests ──────────────────────────────────────

#[test]
fn compile_all_codegen_regression_tests() {
    compile_dir(&common::tests_dir().join("codegen"));
}

// ─── Benchmark files ───────────────────────────────────────────────

#[test]
fn compile_all_bench_tok_files() {
    compile_dir(&common::tests_dir().join("bench"));
}

// ─── Catch-all for any new .tok files ──────────────────────────────

#[test]
fn compile_all_test_tok_files() {
    compile_dir(&common::tests_dir());
}
