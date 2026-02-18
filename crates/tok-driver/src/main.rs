use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use tok_hir::hir::*;

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
    let tokens = match tok_lexer::lex(source) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Lexer error: {}", e);
            process::exit(1);
        }
    };
    match tok_parser::parser::parse(tokens) {
        Ok(program) => {
            for stmt in &program {
                println!("{:#?}", stmt);
            }
            println!("({} statements)", program.len());
        }
        Err(e) => {
            eprintln!("Parse error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_check(source: &str) {
    let tokens = match tok_lexer::lex(source) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Lexer error: {}", e);
            process::exit(1);
        }
    };
    let program = match tok_parser::parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            process::exit(1);
        }
    };
    let type_info = tok_types::check(&program);
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
    // Lex
    let tokens = match tok_lexer::lex(source) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Lexer error: {}", e);
            process::exit(1);
        }
    };

    // Parse
    let program = match tok_parser::parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            process::exit(1);
        }
    };

    // Type check
    let type_info = tok_types::check(&program);
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
    let hir = resolve_file_imports(hir, &source_dir);

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

    // Clean up .o file
    let _ = fs::remove_file(&obj_path);

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

// ─── File-based import resolution for compiled mode ───────────────────

/// Check if a path is a file-based import (not a stdlib module name).
fn is_file_import(path: &str) -> bool {
    path.starts_with("./")
        || path.starts_with("../")
        || path.ends_with(".tok")
        || path.contains('/')
}

/// Resolve the actual file path for an import, adding .tok if needed.
fn resolve_import_path(base_dir: &Path, import_path: &str) -> PathBuf {
    let mut path = base_dir.join(import_path);
    if !path.exists() && !import_path.ends_with(".tok") {
        path = base_dir.join(format!("{}.tok", import_path));
    }
    path
}

/// Parse and lower a .tok file into HIR.
fn compile_file_to_hir(file_path: &Path) -> HirProgram {
    let source = fs::read_to_string(file_path).unwrap_or_else(|e| {
        eprintln!("Error reading imported file {}: {}", file_path.display(), e);
        process::exit(1);
    });
    let tokens = tok_lexer::lex(&source).unwrap_or_else(|e| {
        eprintln!("Lexer error in {}: {}", file_path.display(), e);
        process::exit(1);
    });
    let program = tok_parser::parser::parse(tokens).unwrap_or_else(|e| {
        eprintln!("Parse error in {}: {}", file_path.display(), e);
        process::exit(1);
    });
    let type_info = tok_types::check(&program);
    tok_hir::lower::lower(&program, &type_info)
}

/// Extract exported names and their types from an HIR program.
/// Exports are top-level function declarations and variable assignments
/// whose names don't start with `_`.
fn extract_exports(hir: &HirProgram) -> Vec<(String, tok_types::Type)> {
    let mut exports = Vec::new();
    for stmt in hir {
        match stmt {
            HirStmt::FuncDecl {
                name,
                params,
                ret_type,
                ..
            } if !name.starts_with('_') => {
                let param_tys: Vec<tok_types::ParamType> = params
                    .iter()
                    .map(|p| tok_types::ParamType {
                        ty: p.ty.clone(),
                        has_default: false,
                    })
                    .collect();
                let func_ty = tok_types::Type::Func(tok_types::FuncType {
                    params: param_tys,
                    ret: Box::new(ret_type.clone()),
                    variadic: false,
                });
                exports.push((name.clone(), func_ty));
            }
            HirStmt::Assign { name, ty, .. } if !name.starts_with('_') => {
                exports.push((name.clone(), ty.clone()));
            }
            _ => {}
        }
    }
    exports
}

/// Rename all top-level declarations in an HIR program with a prefix.
/// Also renames references within function bodies to match.
fn prefix_hir_names(hir: &mut HirProgram, prefix: &str, exported: &[(String, tok_types::Type)]) {
    let names_set: HashSet<&str> = exported.iter().map(|(s, _)| s.as_str()).collect();
    for stmt in hir.iter_mut() {
        match stmt {
            HirStmt::FuncDecl { name, body, .. } => {
                if names_set.contains(name.as_str()) {
                    let old_name = name.clone();
                    *name = format!("{}{}", prefix, old_name);
                    // Rename self-references in body (recursion)
                    rename_in_stmts(body, &old_name, name);
                }
                // Rename references to other exported names within this function
                for (export_name, _) in exported {
                    if *export_name != *name && names_set.contains(export_name.as_str()) {
                        let new = format!("{}{}", prefix, export_name);
                        rename_in_stmts(body, export_name, &new);
                    }
                }
            }
            HirStmt::Assign { name, value, .. } => {
                if names_set.contains(name.as_str()) {
                    *name = format!("{}{}", prefix, name);
                }
                // Rename references in the value expression
                for (export_name, _) in exported {
                    let new = format!("{}{}", prefix, export_name);
                    rename_in_expr(value, export_name, &new);
                }
            }
            HirStmt::Expr(expr) => {
                for (export_name, _) in exported {
                    let new = format!("{}{}", prefix, export_name);
                    rename_in_expr(expr, export_name, &new);
                }
            }
            _ => {}
        }
    }
}

fn rename_in_stmts(stmts: &mut [HirStmt], old: &str, new: &str) {
    for stmt in stmts.iter_mut() {
        match stmt {
            HirStmt::Assign { name, value, .. } => {
                if name == old {
                    *name = new.to_string();
                }
                rename_in_expr(value, old, new);
            }
            HirStmt::FuncDecl { name, body, .. } => {
                if name == old {
                    *name = new.to_string();
                }
                rename_in_stmts(body, old, new);
            }
            HirStmt::Expr(expr) => rename_in_expr(expr, old, new),
            HirStmt::Return(Some(expr)) => rename_in_expr(expr, old, new),
            HirStmt::IndexAssign {
                target,
                index,
                value,
            } => {
                rename_in_expr(target, old, new);
                rename_in_expr(index, old, new);
                rename_in_expr(value, old, new);
            }
            HirStmt::MemberAssign { target, value, .. } => {
                rename_in_expr(target, old, new);
                rename_in_expr(value, old, new);
            }
            _ => {}
        }
    }
}

fn rename_in_expr(expr: &mut HirExpr, old: &str, new: &str) {
    match &mut expr.kind {
        HirExprKind::Ident(name) => {
            if name == old {
                *name = new.to_string();
            }
        }
        HirExprKind::BinOp { left, right, .. } => {
            rename_in_expr(left, old, new);
            rename_in_expr(right, old, new);
        }
        HirExprKind::UnaryOp { operand, .. } => {
            rename_in_expr(operand, old, new);
        }
        HirExprKind::Index { target, index } => {
            rename_in_expr(target, old, new);
            rename_in_expr(index, old, new);
        }
        HirExprKind::Member { target, .. } => {
            rename_in_expr(target, old, new);
        }
        HirExprKind::Call { func, args } => {
            rename_in_expr(func, old, new);
            for arg in args.iter_mut() {
                rename_in_expr(arg, old, new);
            }
        }
        HirExprKind::RuntimeCall { args, .. } => {
            for arg in args.iter_mut() {
                rename_in_expr(arg, old, new);
            }
        }
        HirExprKind::Array(elems) => {
            for e in elems.iter_mut() {
                rename_in_expr(e, old, new);
            }
        }
        HirExprKind::Map(entries) => {
            for (_, v) in entries.iter_mut() {
                rename_in_expr(v, old, new);
            }
        }
        HirExprKind::Tuple(elems) => {
            for e in elems.iter_mut() {
                rename_in_expr(e, old, new);
            }
        }
        HirExprKind::Lambda { body, .. } => {
            rename_in_stmts(body, old, new);
        }
        HirExprKind::If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            rename_in_expr(cond, old, new);
            rename_in_stmts(then_body, old, new);
            if let Some(e) = then_expr {
                rename_in_expr(e, old, new);
            }
            rename_in_stmts(else_body, old, new);
            if let Some(e) = else_expr {
                rename_in_expr(e, old, new);
            }
        }
        HirExprKind::Loop { kind, body } => {
            match kind.as_mut() {
                HirLoopKind::While(cond) => rename_in_expr(cond, old, new),
                HirLoopKind::ForRange { start, end, .. } => {
                    rename_in_expr(start, old, new);
                    rename_in_expr(end, old, new);
                }
                HirLoopKind::ForEach { iter, .. } => rename_in_expr(iter, old, new),
                HirLoopKind::ForEachIndexed { iter, .. } => rename_in_expr(iter, old, new),
                HirLoopKind::Infinite => {}
            }
            rename_in_stmts(body, old, new);
        }
        HirExprKind::Block { stmts, expr } => {
            rename_in_stmts(stmts, old, new);
            if let Some(e) = expr {
                rename_in_expr(e, old, new);
            }
        }
        HirExprKind::Length(inner) => rename_in_expr(inner, old, new),
        HirExprKind::Range { start, end, .. } => {
            rename_in_expr(start, old, new);
            rename_in_expr(end, old, new);
        }
        HirExprKind::Go(inner) => rename_in_expr(inner, old, new),
        HirExprKind::Receive(inner) => rename_in_expr(inner, old, new),
        HirExprKind::Send { chan, value } => {
            rename_in_expr(chan, old, new);
            rename_in_expr(value, old, new);
        }
        HirExprKind::Select(arms) => {
            for arm in arms.iter_mut() {
                match arm {
                    HirSelectArm::Recv { chan, body, .. } => {
                        rename_in_expr(chan, old, new);
                        rename_in_stmts(body, old, new);
                    }
                    HirSelectArm::Send { chan, value, body } => {
                        rename_in_expr(chan, old, new);
                        rename_in_expr(value, old, new);
                        rename_in_stmts(body, old, new);
                    }
                    HirSelectArm::Default(body) => {
                        rename_in_stmts(body, old, new);
                    }
                }
            }
        }
        // Literals don't contain references
        HirExprKind::Int(_)
        | HirExprKind::Float(_)
        | HirExprKind::Str(_)
        | HirExprKind::Bool(_)
        | HirExprKind::Nil => {}
    }
}

/// Resolve all file-based imports in the HIR, inlining imported code.
fn resolve_file_imports(mut program: HirProgram, source_dir: &Path) -> HirProgram {
    let mut mod_counter: u32 = 0;
    let mut loaded: HashSet<PathBuf> = HashSet::new();
    let mut preamble: Vec<HirStmt> = Vec::new();

    // Walk the program and resolve imports
    let mut new_program = Vec::new();
    for stmt in program.drain(..) {
        match &stmt {
            // Bare file import: @"./file.tok" → merge exports into scope
            HirStmt::Import(path) if is_file_import(path) => {
                let file_path = resolve_import_path(source_dir, path);
                let canonical = file_path
                    .canonicalize()
                    .unwrap_or_else(|_| file_path.clone());
                let prefix = format!("__mod{}_", mod_counter);
                mod_counter += 1;

                if !loaded.contains(&canonical) {
                    loaded.insert(canonical.clone());
                    let import_dir = file_path.parent().unwrap_or(Path::new(".")).to_path_buf();
                    let mut imported_hir = compile_file_to_hir(&file_path);
                    // Recursively resolve imports in the imported file
                    imported_hir = resolve_file_imports(imported_hir, &import_dir);
                    let exports = extract_exports(&imported_hir);
                    prefix_hir_names(&mut imported_hir, &prefix, &exports);
                    // Add imported declarations to preamble
                    preamble.extend(imported_hir);
                    // Create aliases: export_name = __mod0_export_name
                    for (name, ty) in &exports {
                        let prefixed = format!("{}{}", prefix, name);
                        new_program.push(HirStmt::Assign {
                            name: name.clone(),
                            ty: ty.clone(),
                            value: HirExpr::new(HirExprKind::Ident(prefixed), ty.clone()),
                        });
                    }
                } else {
                    // Already loaded — just create aliases using the existing prefix
                    let imported_hir = compile_file_to_hir(&file_path);
                    let exports = extract_exports(&imported_hir);
                    let existing_prefix = find_existing_prefix(&preamble, &exports);
                    if let Some(ep) = existing_prefix {
                        for (name, ty) in &exports {
                            let prefixed = format!("{}{}", ep, name);
                            new_program.push(HirStmt::Assign {
                                name: name.clone(),
                                ty: ty.clone(),
                                value: HirExpr::new(HirExprKind::Ident(prefixed), ty.clone()),
                            });
                        }
                    }
                }
            }
            _ => {
                // Walk expressions within this statement to find expression-level imports
                let transformed = transform_stmt_imports(
                    stmt,
                    source_dir,
                    &mut mod_counter,
                    &mut loaded,
                    &mut preamble,
                );
                new_program.push(transformed);
            }
        }
    }

    // Prepend preamble (imported functions/vars) before main statements
    let mut result = preamble;
    result.extend(new_program);
    result
}

/// Find the prefix used for a previously-loaded module by checking the preamble.
fn find_existing_prefix(
    preamble: &[HirStmt],
    exports: &[(String, tok_types::Type)],
) -> Option<String> {
    if let Some((first_export, _)) = exports.first() {
        for stmt in preamble {
            if let HirStmt::FuncDecl { name, .. } | HirStmt::Assign { name, .. } = stmt {
                if name.ends_with(first_export.as_str()) && name.starts_with("__mod") {
                    let prefix = &name[..name.len() - first_export.len()];
                    return Some(prefix.to_string());
                }
            }
        }
    }
    None
}

/// Transform a statement, replacing expression-level file imports with map construction.
fn transform_stmt_imports(
    stmt: HirStmt,
    source_dir: &Path,
    mod_counter: &mut u32,
    loaded: &mut HashSet<PathBuf>,
    preamble: &mut Vec<HirStmt>,
) -> HirStmt {
    match stmt {
        HirStmt::Assign {
            name,
            ty,
            mut value,
        } => {
            transform_expr_imports(&mut value, source_dir, mod_counter, loaded, preamble);
            HirStmt::Assign { name, ty, value }
        }
        HirStmt::Expr(mut expr) => {
            transform_expr_imports(&mut expr, source_dir, mod_counter, loaded, preamble);
            HirStmt::Expr(expr)
        }
        HirStmt::FuncDecl {
            name,
            params,
            ret_type,
            mut body,
        } => {
            for s in body.iter_mut() {
                *s = transform_stmt_imports(s.clone(), source_dir, mod_counter, loaded, preamble);
            }
            HirStmt::FuncDecl {
                name,
                params,
                ret_type,
                body,
            }
        }
        other => other,
    }
}

/// Transform expression-level file imports: RuntimeCall("tok_import", [Str(path)])
/// becomes a Map literal of the exported names.
fn transform_expr_imports(
    expr: &mut HirExpr,
    source_dir: &Path,
    mod_counter: &mut u32,
    loaded: &mut HashSet<PathBuf>,
    preamble: &mut Vec<HirStmt>,
) {
    match &mut expr.kind {
        HirExprKind::RuntimeCall { name, args } if name == "tok_import" => {
            if let Some(HirExpr {
                kind: HirExprKind::Str(path),
                ..
            }) = args.first()
            {
                if is_file_import(path) {
                    let path = path.clone();
                    let file_path = resolve_import_path(source_dir, &path);
                    let canonical = file_path
                        .canonicalize()
                        .unwrap_or_else(|_| file_path.clone());
                    let prefix = format!("__mod{}_", *mod_counter);
                    *mod_counter += 1;

                    let (exports, use_prefix) = if !loaded.contains(&canonical) {
                        loaded.insert(canonical.clone());
                        let import_dir = file_path.parent().unwrap_or(Path::new(".")).to_path_buf();
                        let mut imported_hir = compile_file_to_hir(&file_path);
                        imported_hir = resolve_file_imports(imported_hir, &import_dir);
                        let exports = extract_exports(&imported_hir);
                        prefix_hir_names(&mut imported_hir, &prefix, &exports);
                        preamble.extend(imported_hir);
                        (exports, prefix)
                    } else {
                        let imported_hir = compile_file_to_hir(&file_path);
                        let exports = extract_exports(&imported_hir);
                        let ep = find_existing_prefix(preamble, &exports)
                            .unwrap_or_else(|| prefix.clone());
                        (exports, ep)
                    };

                    // Replace tok_import call with a Map literal:
                    // {name1: __mod0_name1, name2: __mod0_name2, ...}
                    let map_entries: Vec<(String, HirExpr)> = exports
                        .iter()
                        .map(|(name, ty)| {
                            let prefixed = format!("{}{}", use_prefix, name);
                            (
                                name.clone(),
                                HirExpr::new(HirExprKind::Ident(prefixed), ty.clone()),
                            )
                        })
                        .collect();
                    expr.kind = HirExprKind::Map(map_entries);
                    expr.ty = tok_types::Type::Map(Box::new(tok_types::Type::Any));
                }
            }
        }
        // Recurse into sub-expressions
        HirExprKind::BinOp { left, right, .. } => {
            transform_expr_imports(left, source_dir, mod_counter, loaded, preamble);
            transform_expr_imports(right, source_dir, mod_counter, loaded, preamble);
        }
        HirExprKind::UnaryOp { operand, .. } => {
            transform_expr_imports(operand, source_dir, mod_counter, loaded, preamble);
        }
        HirExprKind::Call { func, args } => {
            transform_expr_imports(func, source_dir, mod_counter, loaded, preamble);
            for a in args.iter_mut() {
                transform_expr_imports(a, source_dir, mod_counter, loaded, preamble);
            }
        }
        HirExprKind::RuntimeCall { args, .. } => {
            for a in args.iter_mut() {
                transform_expr_imports(a, source_dir, mod_counter, loaded, preamble);
            }
        }
        HirExprKind::Index { target, index } => {
            transform_expr_imports(target, source_dir, mod_counter, loaded, preamble);
            transform_expr_imports(index, source_dir, mod_counter, loaded, preamble);
        }
        HirExprKind::Member { target, .. } => {
            transform_expr_imports(target, source_dir, mod_counter, loaded, preamble);
        }
        HirExprKind::Array(elems) | HirExprKind::Tuple(elems) => {
            for e in elems.iter_mut() {
                transform_expr_imports(e, source_dir, mod_counter, loaded, preamble);
            }
        }
        HirExprKind::Map(entries) => {
            for (_, v) in entries.iter_mut() {
                transform_expr_imports(v, source_dir, mod_counter, loaded, preamble);
            }
        }
        HirExprKind::If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            transform_expr_imports(cond, source_dir, mod_counter, loaded, preamble);
            for s in then_body.iter_mut() {
                *s = transform_stmt_imports(s.clone(), source_dir, mod_counter, loaded, preamble);
            }
            if let Some(e) = then_expr {
                transform_expr_imports(e, source_dir, mod_counter, loaded, preamble);
            }
            for s in else_body.iter_mut() {
                *s = transform_stmt_imports(s.clone(), source_dir, mod_counter, loaded, preamble);
            }
            if let Some(e) = else_expr {
                transform_expr_imports(e, source_dir, mod_counter, loaded, preamble);
            }
        }
        HirExprKind::Lambda { body, .. } => {
            for s in body.iter_mut() {
                *s = transform_stmt_imports(s.clone(), source_dir, mod_counter, loaded, preamble);
            }
        }
        HirExprKind::Loop { kind, body } => {
            match kind.as_mut() {
                HirLoopKind::While(cond) => {
                    transform_expr_imports(cond, source_dir, mod_counter, loaded, preamble);
                }
                HirLoopKind::ForRange { start, end, .. } => {
                    transform_expr_imports(start, source_dir, mod_counter, loaded, preamble);
                    transform_expr_imports(end, source_dir, mod_counter, loaded, preamble);
                }
                HirLoopKind::ForEach { iter, .. } | HirLoopKind::ForEachIndexed { iter, .. } => {
                    transform_expr_imports(iter, source_dir, mod_counter, loaded, preamble);
                }
                HirLoopKind::Infinite => {}
            }
            for s in body.iter_mut() {
                *s = transform_stmt_imports(s.clone(), source_dir, mod_counter, loaded, preamble);
            }
        }
        HirExprKind::Block { stmts, expr } => {
            for s in stmts.iter_mut() {
                *s = transform_stmt_imports(s.clone(), source_dir, mod_counter, loaded, preamble);
            }
            if let Some(e) = expr {
                transform_expr_imports(e, source_dir, mod_counter, loaded, preamble);
            }
        }
        HirExprKind::Length(inner) | HirExprKind::Go(inner) | HirExprKind::Receive(inner) => {
            transform_expr_imports(inner, source_dir, mod_counter, loaded, preamble);
        }
        HirExprKind::Range { start, end, .. } => {
            transform_expr_imports(start, source_dir, mod_counter, loaded, preamble);
            transform_expr_imports(end, source_dir, mod_counter, loaded, preamble);
        }
        HirExprKind::Send { chan, value } => {
            transform_expr_imports(chan, source_dir, mod_counter, loaded, preamble);
            transform_expr_imports(value, source_dir, mod_counter, loaded, preamble);
        }
        HirExprKind::Select(arms) => {
            for arm in arms.iter_mut() {
                match arm {
                    HirSelectArm::Recv { chan, body, .. } => {
                        transform_expr_imports(chan, source_dir, mod_counter, loaded, preamble);
                        for s in body.iter_mut() {
                            *s = transform_stmt_imports(
                                s.clone(),
                                source_dir,
                                mod_counter,
                                loaded,
                                preamble,
                            );
                        }
                    }
                    HirSelectArm::Send { chan, value, body } => {
                        transform_expr_imports(chan, source_dir, mod_counter, loaded, preamble);
                        transform_expr_imports(value, source_dir, mod_counter, loaded, preamble);
                        for s in body.iter_mut() {
                            *s = transform_stmt_imports(
                                s.clone(),
                                source_dir,
                                mod_counter,
                                loaded,
                                preamble,
                            );
                        }
                    }
                    HirSelectArm::Default(body) => {
                        for s in body.iter_mut() {
                            *s = transform_stmt_imports(
                                s.clone(),
                                source_dir,
                                mod_counter,
                                loaded,
                                preamble,
                            );
                        }
                    }
                }
            }
        }
        // Leaves
        HirExprKind::Int(_)
        | HirExprKind::Float(_)
        | HirExprKind::Str(_)
        | HirExprKind::Bool(_)
        | HirExprKind::Nil
        | HirExprKind::Ident(_) => {}
    }
}

fn find_runtime_lib() -> String {
    // Look for libtok_runtime.a in target/debug or target/release
    let candidates = [
        "target/debug/libtok_runtime.a",
        "target/release/libtok_runtime.a",
        // Also check relative to the executable
    ];
    for c in &candidates {
        if Path::new(c).exists() {
            return c.to_string();
        }
    }
    // Try using the cargo-built location from the workspace root
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
