/// Integration tests for the tok driver: full pipeline including import resolution.
///
/// These tests invoke the `tok` binary and validate its behavior end-to-end.
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for unique temp names.
static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tok_binary() -> PathBuf {
    // The tok binary is built alongside tests in target/debug/
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .parent() // crates/
        .unwrap()
        .parent() // workspace root
        .unwrap()
        .join("target")
        .join("debug")
        .join("tok")
}

fn tests_dir() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests")
}

/// Unique temp path for build output to avoid race conditions in parallel tests.
fn temp_output() -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    env::temp_dir().join(format!("tok_driver_test_{}", id))
}

/// Run `tok build <file> -o <output>` and check it succeeds.
fn build_file(path: &Path) -> PathBuf {
    let tok = tok_binary();
    assert!(
        tok.exists(),
        "tok binary not found at {:?}. Run `cargo build` first.",
        tok
    );

    let output_path = temp_output();
    let result = Command::new(&tok)
        .args([
            "build",
            path.to_str().unwrap(),
            "-o",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run tok build");

    assert!(
        result.status.success(),
        "tok build failed for {}:\nstderr: {}",
        path.display(),
        String::from_utf8_lossy(&result.stderr)
    );

    output_path
}

/// Run `tok run <file> -o <temp>` and return (stdout, stderr, exit_code).
fn run_file(path: &Path) -> (String, String, i32) {
    let tok = tok_binary();
    assert!(
        tok.exists(),
        "tok binary not found at {:?}. Run `cargo build` first.",
        tok
    );

    let output_path = temp_output();
    let result = Command::new(&tok)
        .args([
            "run",
            path.to_str().unwrap(),
            "-o",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run tok run");

    let stdout = String::from_utf8_lossy(&result.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&result.stderr).into_owned();
    let code = result.status.code().unwrap_or(-1);
    // Clean up temp binary if it still exists (tok run should clean it, but just in case)
    let _ = std::fs::remove_file(&output_path);
    (stdout, stderr, code)
}

// ─── Build tests ───────────────────────────────────────────────────

#[test]
fn driver_build_basics() {
    let path = tests_dir().join("basics_compile.tok");
    let exe = build_file(&path);
    let _ = std::fs::remove_file(&exe);
}

#[test]
fn driver_build_coverage() {
    let path = tests_dir().join("coverage_test.tok");
    let exe = build_file(&path);
    let _ = std::fs::remove_file(&exe);
}

// ─── Run tests ─────────────────────────────────────────────────────

#[test]
fn driver_run_basics() {
    let path = tests_dir().join("basics_compile.tok");
    let (stdout, stderr, code) = run_file(&path);
    assert_eq!(
        code, 0,
        "non-zero exit code for basics_compile.tok\nstderr: {}",
        stderr
    );
    assert!(
        stdout.contains("30"),
        "expected output '30' in basics_compile.tok"
    );
}

#[test]
fn driver_run_hello() {
    let path = tests_dir().join("hello.tok");
    let (stdout, stderr, code) = run_file(&path);
    assert_eq!(code, 0, "stderr: {}", stderr);
    assert!(
        stdout.contains("Hello World"),
        "expected 'Hello World' in hello.tok output"
    );
}

// ─── Benchmark compilation tests ───────────────────────────────────

#[test]
fn driver_build_all_benchmarks() {
    let bench_dir = tests_dir().join("bench");
    if !bench_dir.exists() {
        return;
    }
    for entry in std::fs::read_dir(&bench_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "tok") {
            let exe = build_file(&path);
            let _ = std::fs::remove_file(&exe);
        }
    }
}

// ─── Codegen regression tests via driver ───────────────────────────

#[test]
fn driver_run_codegen_regression_tests() {
    let codegen_dir = tests_dir().join("codegen");
    if !codegen_dir.exists() {
        return;
    }
    for entry in std::fs::read_dir(&codegen_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "tok") {
            let (_, stderr, code) = run_file(&path);
            assert_eq!(
                code,
                0,
                "codegen regression test {} failed with exit code {}\nstderr: {}",
                path.display(),
                code,
                stderr
            );
        }
    }
}

// ─── Pipeline stage tests ──────────────────────────────────────────

#[test]
fn driver_lex_basics() {
    let tok = tok_binary();
    let path = tests_dir().join("hello.tok");
    let result = Command::new(&tok)
        .args(["lex", path.to_str().unwrap()])
        .output()
        .expect("failed to run tok lex");
    assert!(result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("tokens"),
        "expected token count in lex output"
    );
}

#[test]
fn driver_parse_basics() {
    let tok = tok_binary();
    let path = tests_dir().join("hello.tok");
    let result = Command::new(&tok)
        .args(["parse", path.to_str().unwrap()])
        .output()
        .expect("failed to run tok parse");
    assert!(result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("statements"),
        "expected statement count in parse output"
    );
}

#[test]
fn driver_check_basics() {
    let tok = tok_binary();
    let path = tests_dir().join("hello.tok");
    let result = Command::new(&tok)
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("failed to run tok check");
    assert!(result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("Type check passed"),
        "expected type check pass"
    );
}
