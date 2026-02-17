// Integration test: parse all .tok test files
use std::fs;
use std::path::Path;

fn parse_file(path: &str) {
    let full_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(path);
    let source = fs::read_to_string(&full_path)
        .unwrap_or_else(|e| panic!("Could not read {}: {}", full_path.display(), e));
    let tokens = tok_lexer::lex(&source).unwrap_or_else(|e| panic!("Lex error in {}: {}", path, e));
    let _program = tok_parser::parser::parse(tokens)
        .unwrap_or_else(|e| panic!("Parse error in {}: {}", path, e));
}

#[test]
fn parse_basics() {
    parse_file("tests/basics_test.tok");
}
#[test]
fn parse_control_flow() {
    parse_file("tests/control_flow_test.tok");
}
#[test]
fn parse_functions() {
    parse_file("tests/functions_test.tok");
}
#[test]
fn parse_arrays_lambdas() {
    parse_file("tests/arrays_lambdas_test.tok");
}
#[test]
fn parse_strings_pipes() {
    parse_file("tests/strings_pipes_test.tok");
}
#[test]
fn parse_maps() {
    parse_file("tests/maps_test.tok");
}
#[test]
fn parse_math_helpers() {
    parse_file("tests/math_helpers.tok");
}
#[test]
fn parse_string_utils() {
    parse_file("tests/_string_utils.tok");
}
#[test]
fn parse_uses_math() {
    parse_file("tests/_uses_math.tok");
}
#[test]
fn parse_errors_tuples() {
    parse_file("tests/errors_tuples_test.tok");
}
#[test]
fn parse_concurrency() {
    parse_file("tests/concurrency_test.tok");
}
#[test]
fn parse_imports() {
    parse_file("tests/imports_test.tok");
}
#[test]
fn parse_bench_01() {
    parse_file("tests/bench/01_recursion.tok");
}
#[test]
fn parse_bench_02() {
    parse_file("tests/bench/02_loops.tok");
}
#[test]
fn parse_bench_03() {
    parse_file("tests/bench/03_arrays.tok");
}
#[test]
fn parse_bench_04() {
    parse_file("tests/bench/04_strings.tok");
}
#[test]
fn parse_bench_05() {
    parse_file("tests/bench/05_maps.tok");
}
#[test]
fn parse_bench_06() {
    parse_file("tests/bench/06_closures.tok");
}
#[test]
fn parse_bench_07() {
    parse_file("tests/bench/07_concurrency.tok");
}
#[test]
fn parse_bench_08() {
    parse_file("tests/bench/08_combined.tok");
}
