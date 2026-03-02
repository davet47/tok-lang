// ─── Shared test helpers for tok-codegen integration tests ──────────

use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for unique temp file names across threads.
static COUNTER: AtomicU64 = AtomicU64::new(0);

// ─── Pipeline: source → object bytes ───────────────────────────────

/// Run the full compilation pipeline: lex → parse → typecheck → HIR lower → codegen.
/// Returns the raw object file bytes (`.o` format).
/// Panics on lex/parse errors (test failure).
pub fn compile_source(source: &str) -> Vec<u8> {
    let tokens = tok_lexer::lex(source).expect("lex error");
    let ast = tok_parser::parser::parse(tokens).expect("parse error");
    let type_info = tok_types::check(&ast);
    let hir = tok_hir::lower::lower(&ast, &type_info);
    tok_codegen::compile(&hir)
}

// ─── Pipeline: source → executable → run ───────────────────────────

/// Compile a Tok source string to a native executable, run it, and return
/// `(stdout, stderr, exit_code)`.
///
/// Uses a unique temp directory per call to avoid collisions when tests
/// run in parallel.
pub fn compile_and_run(source: &str) -> (String, String, i32) {
    let obj_bytes = compile_source(source);

    // Create unique temp dir
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let tmp_dir = env::temp_dir().join(format!("tok_test_{}", id));
    fs::create_dir_all(&tmp_dir).expect("failed to create temp dir");

    let obj_path = tmp_dir.join("test.o");
    let exe_path = tmp_dir.join("test_exe");

    fs::write(&obj_path, &obj_bytes).expect("failed to write object file");

    // Link with runtime
    let runtime_lib = find_runtime_lib();
    let link_status = Command::new("cc")
        .args([
            obj_path.to_str().unwrap(),
            &runtime_lib,
            "-o",
            exe_path.to_str().unwrap(),
            "-lpthread",
        ])
        .output()
        .expect("failed to run linker (cc)");

    if !link_status.status.success() {
        let stderr = String::from_utf8_lossy(&link_status.stderr);
        // Clean up
        let _ = fs::remove_dir_all(&tmp_dir);
        panic!("Linker failed: {}", stderr);
    }

    // Run the executable
    let output = Command::new(&exe_path)
        .output()
        .expect("failed to run compiled executable");

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let exit_code = output.status.code().unwrap_or(-1);

    // Clean up temp directory
    let _ = fs::remove_dir_all(&tmp_dir);

    (stdout, stderr, exit_code)
}

// ─── Compile + run a .tok file by path ─────────────────────────────

/// Read a `.tok` file, compile it, run it, and validate against `// expect:` annotations.
/// Panics (fails the test) if output doesn't match expectations.
pub fn compile_run_and_validate_file(path: &Path) {
    let source = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));

    let (stdout, stderr, exit_code) = compile_and_run(&source);

    // Check exit code
    assert_eq!(
        exit_code, 0,
        "non-zero exit code ({}) for {}\nstderr: {}",
        exit_code,
        path.display(),
        stderr
    );

    // Validate output against // expect: annotations
    let expected = parse_expected_output(&source);
    if !expected.is_empty() {
        let actual_lines: Vec<&str> = stdout.lines().collect();
        for (i, exp) in expected.iter().enumerate() {
            let actual = actual_lines.get(i).unwrap_or(&"<missing>");
            assert_eq!(
                actual, exp,
                "output mismatch in {} at line {}: expected {:?}, got {:?}\n\nFull output:\n{}",
                path.display(),
                i + 1,
                exp,
                actual,
                stdout
            );
        }
        // Also check we didn't get extra unexpected output
        if actual_lines.len() > expected.len() {
            panic!(
                "extra output in {}: expected {} lines, got {}\n\nExpected:\n{}\n\nActual:\n{}",
                path.display(),
                expected.len(),
                actual_lines.len(),
                expected.join("\n"),
                stdout
            );
        }
    }
}

// ─── Parse // expect: annotations ──────────────────────────────────

/// Extract expected output lines from `// expect:` comments in source.
///
/// Format:
/// - `// expect: VALUE` → one expected line "VALUE"
/// - `// expect:` (empty) followed by consecutive `// ...` lines → multi-line expected output
pub fn parse_expected_output(source: &str) -> Vec<String> {
    let mut expected = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        if let Some(rest) = trimmed.strip_prefix("// expect:") {
            let rest = rest.trim();
            if rest.is_empty() {
                // Multi-line: consume consecutive // lines
                i += 1;
                while i < lines.len() {
                    let next = lines[i].trim();
                    if let Some(content) = next.strip_prefix("//") {
                        // Strip one leading space if present
                        let content = content.strip_prefix(' ').unwrap_or(content);
                        expected.push(content.to_string());
                        i += 1;
                    } else {
                        break;
                    }
                }
                continue; // Don't increment i again
            } else {
                expected.push(rest.to_string());
            }
        }
        i += 1;
    }

    expected
}

// ─── Locate runtime library ────────────────────────────────────────

/// Find `libtok_runtime.a` for linking. Searches:
/// 1. Workspace target/debug/
/// 2. Workspace target/release/
/// 3. Relative to CARGO_MANIFEST_DIR
fn find_runtime_lib() -> String {
    let lib_name = "libtok_runtime.a";

    // Try workspace root target directories
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir)
        .parent() // crates/
        .and_then(|p| p.parent()); // workspace root

    if let Some(root) = workspace_root {
        for profile in &["debug", "release"] {
            let candidate = root.join("target").join(profile).join(lib_name);
            if candidate.exists() {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }

    // Try CWD-relative
    for profile in &["debug", "release"] {
        let candidate = Path::new("target").join(profile).join(lib_name);
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }

    panic!(
        "Could not find {}. Run `cargo build -p tok-runtime` first.",
        lib_name
    );
}

// ─── Test directory helpers ────────────────────────────────────────

/// Get the path to the workspace `tests/` directory.
pub fn tests_dir() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .parent() // crates/
        .unwrap()
        .parent() // workspace root
        .unwrap()
        .join("tests")
}

/// Check if a .tok file should be skipped for compile+run tests.
pub fn should_skip_run(filename: &str) -> bool {
    let skip_list = [
        "stdlib_io_test.tok",   // reads stdin
        "stdlib_http_test.tok", // network requests
        "stdlib_llm_test.tok",  // needs API key
        "imports_test.tok",     // needs import resolver
        "stdlib_tmpl_test.tok", // needs file I/O for templates
    ];
    // Skip private modules and debug files
    filename.starts_with('_')
        || filename.starts_with("debug")
        || skip_list.contains(&filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_expected_single_line() {
        let source = r#"
pl(42)
// expect: 42
pl("hello")
// expect: hello
"#;
        let expected = parse_expected_output(source);
        assert_eq!(expected, vec!["42", "hello"]);
    }

    #[test]
    fn test_parse_expected_multiline() {
        let source = r#"
~(i:0..3){pl(i)}
// expect:
// 0
// 1
// 2
"#;
        let expected = parse_expected_output(source);
        assert_eq!(expected, vec!["0", "1", "2"]);
    }

    #[test]
    fn test_parse_expected_mixed() {
        let source = r#"
pl("start")
// expect: start
~(i:0..2){pl(i)}
// expect:
// 0
// 1
pl("end")
// expect: end
"#;
        let expected = parse_expected_output(source);
        assert_eq!(expected, vec!["start", "0", "1", "end"]);
    }

    #[test]
    fn test_parse_expected_empty() {
        let source = "pl(42)\n// no expect here\n";
        let expected = parse_expected_output(source);
        assert!(expected.is_empty());
    }
}
