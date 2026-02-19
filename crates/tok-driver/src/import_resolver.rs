//! File-based import resolution for compiled mode.
//!
//! Resolves `@"./file.tok"` imports by parsing imported files,
//! prefixing their names, and inlining them into the main HIR program.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process;

use tok_hir::hir::*;
use tok_types::Type;

/// Cached info for a previously-loaded module.
struct ModuleCache {
    prefix: String,
    exports: Vec<(String, Type)>,
}

/// State threaded through the import resolution pass.
struct ImportCtx {
    mod_counter: u32,
    loaded: HashMap<PathBuf, ModuleCache>,
    /// Stack of files currently being imported, for cycle detection.
    import_stack: Vec<PathBuf>,
    preamble: Vec<HirStmt>,
}

impl ImportCtx {
    fn new() -> Self {
        ImportCtx {
            mod_counter: 0,
            loaded: HashMap::new(),
            import_stack: Vec::new(),
            preamble: Vec::new(),
        }
    }

    fn next_prefix(&mut self) -> String {
        let prefix = format!("__mod{}_", self.mod_counter);
        self.mod_counter += 1;
        prefix
    }

    /// Load a file-based module, returning its (exports, prefix).
    /// Uses caching so each file is compiled at most once.
    /// Detects circular imports and exits with an error message.
    fn load_module(
        &mut self,
        source_dir: &Path,
        import_path: &str,
    ) -> (Vec<(String, Type)>, String) {
        let file_path = resolve_import_path(source_dir, import_path);
        let canonical = file_path
            .canonicalize()
            .unwrap_or_else(|_| file_path.clone());

        if let Some(cached) = self.loaded.get(&canonical) {
            return (cached.exports.clone(), cached.prefix.clone());
        }

        // Circular import detection
        if self.import_stack.contains(&canonical) {
            let cycle: Vec<String> = self
                .import_stack
                .iter()
                .map(|p| p.display().to_string())
                .collect();
            eprintln!(
                "Circular import detected: {} -> {}",
                cycle.join(" -> "),
                canonical.display()
            );
            process::exit(1);
        }

        let prefix = self.next_prefix();
        self.import_stack.push(canonical.clone());

        let import_dir = file_path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let mut imported_hir = compile_file_to_hir(&file_path);
        imported_hir = resolve_file_imports_inner(imported_hir, &import_dir, self);
        let exports = extract_exports(&imported_hir);
        prefix_hir_names(&mut imported_hir, &prefix, &exports);
        self.preamble.extend(imported_hir);

        self.import_stack.pop();
        self.loaded.insert(
            canonical,
            ModuleCache {
                prefix: prefix.clone(),
                exports: exports.clone(),
            },
        );
        (exports, prefix)
    }
}

/// Known stdlib module names — anything else is a file import.
const STDLIB_MODULES: &[&str] = &[
    "math", "str", "os", "io", "json", "csv", "fs", "http", "re", "time", "tmpl", "toon", "llm",
];

/// Check if a path is a file-based import (not a stdlib module name).
fn is_file_import(path: &str) -> bool {
    !STDLIB_MODULES.contains(&path)
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
    let source = std::fs::read_to_string(file_path).unwrap_or_else(|e| {
        eprintln!("Error reading imported file {}: {}", file_path.display(), e);
        process::exit(1);
    });
    crate::parse_source_to_hir(&source, Some(file_path))
}

/// Extract exported names and their types from an HIR program.
/// Exports are top-level function declarations and variable assignments
/// whose names don't start with `_`.
fn extract_exports(hir: &HirProgram) -> Vec<(String, Type)> {
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
                let func_ty = Type::Func(tok_types::FuncType {
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

// ─── Generic HIR walker ──────────────────────────────────────────────
//
// A single recursive traversal that supports both renaming and import
// transformation, eliminating the duplicated ~250-line traversals.

/// Callbacks for the HIR walker. Implementations can override only what they need.
trait HirVisitor {
    /// Visit an identifier. Return a new name to rename it, or None to keep it.
    fn visit_ident(&mut self, _name: &str) -> Option<String> {
        None
    }

    /// Visit a RuntimeCall. Return true if the expression was replaced in-place.
    fn visit_runtime_call(&mut self, _expr: &mut HirExpr, _name: &str, _args: &[HirExpr]) -> bool {
        false
    }
}

/// Maximum HIR nesting depth before aborting traversal.
/// Prevents stack overflow on adversarially nested input.
const MAX_WALK_DEPTH: usize = 1000;

fn walk_stmts(stmts: &mut [HirStmt], visitor: &mut dyn HirVisitor, depth: usize) {
    for stmt in stmts.iter_mut() {
        walk_stmt(stmt, visitor, depth);
    }
}

fn walk_stmt(stmt: &mut HirStmt, visitor: &mut dyn HirVisitor, depth: usize) {
    if depth >= MAX_WALK_DEPTH {
        eprintln!("warning: HIR nesting depth exceeded {MAX_WALK_DEPTH}, skipping subtree");
        return;
    }
    match stmt {
        HirStmt::Assign { name, value, .. } => {
            if let Some(new) = visitor.visit_ident(name) {
                *name = new;
            }
            walk_expr(value, visitor, depth + 1);
        }
        HirStmt::FuncDecl { name, body, .. } => {
            if let Some(new) = visitor.visit_ident(name) {
                *name = new;
            }
            walk_stmts(body, visitor, depth + 1);
        }
        HirStmt::Expr(expr) => walk_expr(expr, visitor, depth + 1),
        HirStmt::Return(Some(expr)) => walk_expr(expr, visitor, depth + 1),
        HirStmt::IndexAssign {
            target,
            index,
            value,
        } => {
            walk_expr(target, visitor, depth + 1);
            walk_expr(index, visitor, depth + 1);
            walk_expr(value, visitor, depth + 1);
        }
        HirStmt::MemberAssign { target, value, .. } => {
            walk_expr(target, visitor, depth + 1);
            walk_expr(value, visitor, depth + 1);
        }
        _ => {}
    }
}

fn walk_expr(expr: &mut HirExpr, visitor: &mut dyn HirVisitor, depth: usize) {
    if depth >= MAX_WALK_DEPTH {
        return;
    }
    // First try runtime call interception (before recursing into children).
    // We extract (name, args) to avoid borrow conflict with the mutable `expr`.
    let is_import_call = matches!(
        &expr.kind,
        HirExprKind::RuntimeCall { name, args }
        if name == "tok_import" && !args.is_empty()
    );
    if is_import_call {
        // Extract the info we need, then pass &mut expr
        let (name_clone, args_clone) = match &expr.kind {
            HirExprKind::RuntimeCall { name, args } => (name.clone(), args.clone()),
            _ => unreachable!(),
        };
        if visitor.visit_runtime_call(expr, &name_clone, &args_clone) {
            return; // Expression was replaced in-place
        }
    }

    let d = depth + 1;
    match &mut expr.kind {
        HirExprKind::Ident(name) => {
            if let Some(new) = visitor.visit_ident(name) {
                *name = new;
            }
        }
        HirExprKind::BinOp { left, right, .. } => {
            walk_expr(left, visitor, d);
            walk_expr(right, visitor, d);
        }
        HirExprKind::UnaryOp { operand, .. } => {
            walk_expr(operand, visitor, d);
        }
        HirExprKind::Index { target, index } => {
            walk_expr(target, visitor, d);
            walk_expr(index, visitor, d);
        }
        HirExprKind::Member { target, .. } => {
            walk_expr(target, visitor, d);
        }
        HirExprKind::Call { func, args } => {
            walk_expr(func, visitor, d);
            for arg in args.iter_mut() {
                walk_expr(arg, visitor, d);
            }
        }
        HirExprKind::RuntimeCall { args, .. } => {
            for arg in args.iter_mut() {
                walk_expr(arg, visitor, d);
            }
        }
        HirExprKind::Array(elems) | HirExprKind::Tuple(elems) => {
            for e in elems.iter_mut() {
                walk_expr(e, visitor, d);
            }
        }
        HirExprKind::Map(entries) => {
            for (_, v) in entries.iter_mut() {
                walk_expr(v, visitor, d);
            }
        }
        HirExprKind::Lambda { body, .. } => {
            walk_stmts(body, visitor, d);
        }
        HirExprKind::If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            walk_expr(cond, visitor, d);
            walk_stmts(then_body, visitor, d);
            if let Some(e) = then_expr {
                walk_expr(e, visitor, d);
            }
            walk_stmts(else_body, visitor, d);
            if let Some(e) = else_expr {
                walk_expr(e, visitor, d);
            }
        }
        HirExprKind::Loop { kind, body } => {
            match kind.as_mut() {
                HirLoopKind::While(cond) => walk_expr(cond, visitor, d),
                HirLoopKind::ForRange { start, end, .. } => {
                    walk_expr(start, visitor, d);
                    walk_expr(end, visitor, d);
                }
                HirLoopKind::ForEach { iter, .. } | HirLoopKind::ForEachIndexed { iter, .. } => {
                    walk_expr(iter, visitor, d);
                }
                HirLoopKind::Infinite => {}
            }
            walk_stmts(body, visitor, d);
        }
        HirExprKind::Block { stmts, expr } => {
            walk_stmts(stmts, visitor, d);
            if let Some(e) = expr {
                walk_expr(e, visitor, d);
            }
        }
        HirExprKind::Length(inner) | HirExprKind::Go(inner) | HirExprKind::Receive(inner) => {
            walk_expr(inner, visitor, d);
        }
        HirExprKind::Range { start, end, .. } => {
            walk_expr(start, visitor, d);
            walk_expr(end, visitor, d);
        }
        HirExprKind::Send { chan, value } => {
            walk_expr(chan, visitor, d);
            walk_expr(value, visitor, d);
        }
        HirExprKind::Select(arms) => {
            for arm in arms.iter_mut() {
                match arm {
                    HirSelectArm::Recv { chan, body, .. } => {
                        walk_expr(chan, visitor, d);
                        walk_stmts(body, visitor, d);
                    }
                    HirSelectArm::Send { chan, value, body } => {
                        walk_expr(chan, visitor, d);
                        walk_expr(value, visitor, d);
                        walk_stmts(body, visitor, d);
                    }
                    HirSelectArm::Default(body) => {
                        walk_stmts(body, visitor, d);
                    }
                }
            }
        }
        // Literals don't need visiting
        HirExprKind::Int(_)
        | HirExprKind::Float(_)
        | HirExprKind::Str(_)
        | HirExprKind::Bool(_)
        | HirExprKind::Nil => {}
    }
}

// ─── Rename visitor ──────────────────────────────────────────────────

/// Batch-renames identifiers using a HashMap lookup (single-pass, O(1) per ident).
struct BatchRenamer<'a> {
    renames: &'a HashMap<String, String>,
}

impl<'a> HirVisitor for BatchRenamer<'a> {
    fn visit_ident(&mut self, name: &str) -> Option<String> {
        self.renames.get(name).cloned()
    }
}

/// Rename all top-level declarations in an HIR program with a prefix.
/// Also renames references within function bodies to match.
///
/// Uses a single-pass batch renamer: builds the full old→new map once,
/// then walks each statement body exactly once (O(exports + HIR_size)
/// instead of the old O(exports × HIR_size)).
fn prefix_hir_names(hir: &mut HirProgram, prefix: &str, exported: &[(String, Type)]) {
    // Build rename map: old_name → new_name
    let renames: HashMap<String, String> = exported
        .iter()
        .map(|(name, _)| (name.clone(), format!("{}{}", prefix, name)))
        .collect();
    let mut renamer = BatchRenamer { renames: &renames };

    for stmt in hir.iter_mut() {
        match stmt {
            HirStmt::FuncDecl { name, body, .. } => {
                if let Some(new_name) = renames.get(name.as_str()) {
                    *name = new_name.clone();
                }
                walk_stmts(body, &mut renamer, 0);
            }
            HirStmt::Assign { name, value, .. } => {
                if let Some(new_name) = renames.get(name.as_str()) {
                    *name = new_name.clone();
                }
                walk_expr(value, &mut renamer, 0);
            }
            HirStmt::Expr(expr) => {
                walk_expr(expr, &mut renamer, 0);
            }
            _ => {}
        }
    }
}

// ─── Import transformer visitor ──────────────────────────────────────

/// Transforms expression-level file imports (RuntimeCall("tok_import", [Str(path)]))
/// into Map literals of the exported names, and adds imported code to the preamble.
///
/// This replaces the old separate `transform_expr_imports` traversal.
struct ImportTransformer<'a> {
    source_dir: PathBuf,
    ictx: &'a mut ImportCtx,
}

impl<'a> HirVisitor for ImportTransformer<'a> {
    fn visit_runtime_call(&mut self, expr: &mut HirExpr, name: &str, args: &[HirExpr]) -> bool {
        if name != "tok_import" {
            return false;
        }
        let path = match args.first() {
            Some(HirExpr {
                kind: HirExprKind::Str(p),
                ..
            }) => p.clone(),
            _ => return false,
        };
        if !is_file_import(&path) {
            return false;
        }

        let (exports, use_prefix) = self.ictx.load_module(&self.source_dir, &path);

        // Replace tok_import call with a Map literal
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
        expr.ty = Type::Map(Box::new(Type::Any));
        true
    }
}

// ─── Main import resolution ──────────────────────────────────────────

/// Resolve all file-based imports in the HIR, inlining imported code.
pub fn resolve_file_imports(program: HirProgram, source_dir: &Path) -> HirProgram {
    let mut ictx = ImportCtx::new();
    resolve_file_imports_inner(program, source_dir, &mut ictx)
}

fn resolve_file_imports_inner(
    mut program: HirProgram,
    source_dir: &Path,
    ictx: &mut ImportCtx,
) -> HirProgram {
    let mut new_program = Vec::new();

    for mut stmt in program.drain(..) {
        match &stmt {
            // Bare file import: @"./file.tok" → merge exports into scope
            HirStmt::Import(path) if is_file_import(path) => {
                let (exports, use_prefix) = ictx.load_module(source_dir, path);

                // Create aliases: export_name = __mod0_export_name
                for (name, ty) in &exports {
                    let prefixed = format!("{}{}", use_prefix, name);
                    new_program.push(HirStmt::Assign {
                        name: name.clone(),
                        ty: ty.clone(),
                        value: HirExpr::new(HirExprKind::Ident(prefixed), ty.clone()),
                    });
                }
            }
            _ => {
                // Walk this statement to resolve expression-level imports
                let mut transformer = ImportTransformer {
                    source_dir: source_dir.to_path_buf(),
                    ictx,
                };
                walk_stmt(&mut stmt, &mut transformer, 0);
                new_program.push(stmt);
            }
        }
    }

    // Prepend preamble (imported functions/vars) before main statements
    let mut result = std::mem::take(&mut ictx.preamble);
    result.extend(new_program);
    result
}
