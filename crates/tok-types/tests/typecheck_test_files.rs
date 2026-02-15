// Integration test: parse and type-check all .tok test files
use std::fs;
use std::path::Path;

fn typecheck_file(path: &str) {
    let full_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join(path);
    let source = fs::read_to_string(&full_path)
        .unwrap_or_else(|e| panic!("Could not read {}: {}", full_path.display(), e));
    let tokens = tok_lexer::lex(&source)
        .unwrap_or_else(|e| panic!("Lex error in {}: {}", path, e));
    let program = tok_parser::parser::parse(tokens)
        .unwrap_or_else(|e| panic!("Parse error in {}: {}", path, e));
    let _info = tok_types::check(&program);
    // Success = no panic. Warnings are acceptable for untyped code.
}

#[test] fn typecheck_basics()          { typecheck_file("tests/basics_test.tok"); }
#[test] fn typecheck_control_flow()    { typecheck_file("tests/control_flow_test.tok"); }
#[test] fn typecheck_functions()       { typecheck_file("tests/functions_test.tok"); }
#[test] fn typecheck_arrays_lambdas()  { typecheck_file("tests/arrays_lambdas_test.tok"); }
#[test] fn typecheck_strings_pipes()   { typecheck_file("tests/strings_pipes_test.tok"); }
#[test] fn typecheck_maps()            { typecheck_file("tests/maps_test.tok"); }
#[test] fn typecheck_math_helpers()    { typecheck_file("tests/math_helpers.tok"); }
#[test] fn typecheck_string_utils()    { typecheck_file("tests/_string_utils.tok"); }
#[test] fn typecheck_uses_math()       { typecheck_file("tests/_uses_math.tok"); }
#[test] fn typecheck_errors_tuples()   { typecheck_file("tests/errors_tuples_test.tok"); }
#[test] fn typecheck_concurrency()     { typecheck_file("tests/concurrency_test.tok"); }
#[test] fn typecheck_imports()         { typecheck_file("tests/imports_test.tok"); }
#[test] fn typecheck_bench_01()        { typecheck_file("tests/bench/01_recursion.tok"); }
#[test] fn typecheck_bench_02()        { typecheck_file("tests/bench/02_loops.tok"); }
#[test] fn typecheck_bench_03()        { typecheck_file("tests/bench/03_arrays.tok"); }
#[test] fn typecheck_bench_04()        { typecheck_file("tests/bench/04_strings.tok"); }
#[test] fn typecheck_bench_05()        { typecheck_file("tests/bench/05_maps.tok"); }
#[test] fn typecheck_bench_06()        { typecheck_file("tests/bench/06_closures.tok"); }
#[test] fn typecheck_bench_07()        { typecheck_file("tests/bench/07_concurrency.tok"); }
#[test] fn typecheck_bench_08()        { typecheck_file("tests/bench/08_combined.tok"); }
