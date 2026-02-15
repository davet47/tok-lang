/// Cranelift code generation for the Tok language.
///
/// Takes an HIR program (from `tok_hir::lower`) and compiles it to native
/// machine code via Cranelift, producing a `.o` object file that is then
/// linked with `libtok_rt.a` to produce an executable.

pub mod compiler;

pub use compiler::compile;
