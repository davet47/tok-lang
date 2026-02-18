mod import_resolver;

use std::env;
use std::fs;
use std::path::Path;
use std::process::{self, Command};

use tok_hir::hir::HirProgram;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: tok <command> <file> [-o output]");
        eprintln!("Commands: lex, parse, check, build, run");
        process::exit(1);
    }

    let command = &args[1];
    let file = &args[2];

    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {}", file, e);
            process::exit(1);
        }
    };

    match command.as_str() {
        "lex" => cmd_lex(&source),
        "parse" => cmd_parse(&source),
        "check" => cmd_check(&source),
        "build" => {
            let output = get_output_path(&args, file);
            cmd_build(&source, file, &output);
        }
        "run" => {
            let output = get_output_path(&args, file);
            cmd_build(&source, file, &output);
            cmd_run(&output);
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            process::exit(1);
        }
    }
}

fn get_output_path(args: &[String], input: &str) -> String {
    // Check for -o flag
    for i in 3..args.len() {
        if args[i] == "-o" && i + 1 < args.len() {
            return args[i + 1].clone();
        }
    }
    // Default: strip .tok extension
    let p = Path::new(input);
    let stem = p.file_stem().unwrap_or_default().to_str().unwrap_or("a");
    let parent = p.parent().unwrap_or(Path::new("."));
    parent.join(stem).to_str().unwrap().to_string()
}

// ─── Shared pipeline helpers ─────────────────────────────────────────

/// Lex + parse + type-check a source string, exiting on error.
/// Returns (program, type_info).
fn parse_source(source: &str) -> (Vec<tok_parser::ast::Stmt>, tok_types::TypeInfo) {
    let tokens = tok_lexer::lex(source).unwrap_or_else(|e| {
        eprintln!("Lexer error: {}", e);
        process::exit(1);
    });
    let program = tok_parser::parser::parse(tokens).unwrap_or_else(|e| {
        eprintln!("Parse error: {}", e);
        process::exit(1);
    });
    let type_info = tok_types::check(&program);
    (program, type_info)
}

/// Lex + parse + type-check + lower to HIR.
/// Used by both cmd_build and import_resolver.
pub(crate) fn parse_source_to_hir(source: &str, file_path: Option<&Path>) -> HirProgram {
    let tokens = tok_lexer::lex(source).unwrap_or_else(|e| {
        let ctx = file_path
            .map(|p| format!(" in {}", p.display()))
            .unwrap_or_default();
        eprintln!("Lexer error{}: {}", ctx, e);
        process::exit(1);
    });
    let program = tok_parser::parser::parse(tokens).unwrap_or_else(|e| {
        let ctx = file_path
            .map(|p| format!(" in {}", p.display()))
            .unwrap_or_default();
        eprintln!("Parse error{}: {}", ctx, e);
        process::exit(1);
    });
    let type_info = tok_types::check(&program);
    tok_hir::lower::lower(&program, &type_info)
}

// ─── Commands ────────────────────────────────────────────────────────

fn cmd_lex(source: &str) {
    let tokens = tok_lexer::lex(source);
    match tokens {
        Ok(toks) => {
            for tok in &toks {
                println!("{:?}", tok);
            }
            println!("({} tokens)", toks.len());
        }
        Err(e) => {
            eprintln!("Lexer error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_parse(source: &str) {
    let (program, _) = parse_source(source);
    for stmt in &program {
        println!("{:#?}", stmt);
    }
    println!("({} statements)", program.len());
}

fn cmd_check(source: &str) {
    let (_, type_info) = parse_source(source);
    if type_info.warnings.is_empty() {
        println!("Type check passed (no warnings)");
    } else {
        println!(
            "Type check passed with {} warnings:",
            type_info.warnings.len()
        );
        for w in &type_info.warnings {
            println!("  - {:?}", w);
        }
    }
    println!("Functions: {}", type_info.functions.len());
    println!("Variables: {}", type_info.variables.len());
}

fn cmd_build(source: &str, input_file: &str, output_path: &str) {
    let (program, type_info) = parse_source(source);

    for w in &type_info.warnings {
        eprintln!("warning: {:?}", w);
    }

    // Lower to HIR
    let hir = tok_hir::lower::lower(&program, &type_info);

    // Resolve file-based imports
    let source_dir = Path::new(input_file)
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let hir = import_resolver::resolve_file_imports(hir, &source_dir);

    // Generate object file
    let obj_bytes = tok_codegen::compile(&hir);

    // Write .o file
    let obj_path = format!("{}.o", output_path);
    fs::write(&obj_path, &obj_bytes).unwrap_or_else(|e| {
        eprintln!("Error writing {}: {}", obj_path, e);
        process::exit(1);
    });

    // Find the runtime library
    let runtime_lib = find_runtime_lib();

    // Link
    let status = Command::new("cc")
        .args([&obj_path, &runtime_lib, "-o", output_path, "-lpthread"])
        .status();

    // Clean up .o file, warn on failure
    if let Err(e) = fs::remove_file(&obj_path) {
        eprintln!("Warning: failed to clean up {}: {}", obj_path, e);
    }

    match status {
        Ok(s) if s.success() => {
            eprintln!("Built: {}", output_path);
        }
        Ok(s) => {
            eprintln!("Linker failed with exit code: {}", s);
            process::exit(1);
        }
        Err(e) => {
            eprintln!("Failed to run linker: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_run(output_path: &str) {
    let status = Command::new(output_path).status();
    // Clean up the temporary executable for `run` command
    let _ = fs::remove_file(output_path);
    match status {
        Ok(s) => {
            process::exit(s.code().unwrap_or(1));
        }
        Err(e) => {
            eprintln!("Failed to run {}: {}", output_path, e);
            process::exit(1);
        }
    }
}

fn find_runtime_lib() -> String {
    // Look for libtok_runtime.a in target/debug or target/release
    let candidates = [
        "target/debug/libtok_runtime.a",
        "target/release/libtok_runtime.a",
    ];
    for c in &candidates {
        if Path::new(c).exists() {
            return c.to_string();
        }
    }
    // Try using the cargo-built location relative to the executable
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    if let Some(dir) = exe_dir {
        let lib = dir.join("libtok_runtime.a");
        if lib.exists() {
            return lib.to_str().unwrap().to_string();
        }
    }
    // Fallback
    eprintln!("Warning: could not find libtok_runtime.a, trying system path");
    "libtok_runtime.a".to_string()
}
