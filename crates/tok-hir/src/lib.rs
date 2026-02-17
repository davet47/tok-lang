/// HIR (High-level Intermediate Representation) for the Tok language compiler.
///
/// Provides a desugared, type-annotated IR that is easy for codegen to consume.
/// The `lower` module transforms a parsed AST + TypeInfo into this representation.
pub mod hir;
pub mod lower;
