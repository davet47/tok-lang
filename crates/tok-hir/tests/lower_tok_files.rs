/// Integration tests: lex -> parse -> typecheck -> lower for all .tok test files.
///
/// Ensures the HIR lowering pass does not panic on any real Tok source file.
use std::fs;
use std::path::Path;

fn lower_file(path: &Path) {
    let source = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    let tokens = tok_lexer::lex(&source)
        .unwrap_or_else(|e| panic!("lex error in {}: {:?}", path.display(), e));
    let ast = tok_parser::parser::parse(tokens)
        .unwrap_or_else(|e| panic!("parse error in {}: {:?}", path.display(), e));
    let type_info = tok_types::check(&ast);
    let _hir = tok_hir::lower::lower(&ast, &type_info);
    // If we got here without panicking, the lowering succeeded.
}

fn test_dir(dir: &str) {
    let dir_path = Path::new(dir);
    if !dir_path.exists() {
        return;
    }
    for entry in fs::read_dir(dir_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "tok") {
            lower_file(&path);
        }
    }
}

#[test]
fn lower_all_test_tok_files() {
    test_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests"));
}

#[test]
fn lower_all_bench_tok_files() {
    test_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/bench"));
}
