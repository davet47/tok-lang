//! Tok Language Runtime Library
//!
//! This is the runtime support library for AOT-compiled Tok programs.
//! It provides reference-counted heap types (strings, arrays, maps, tuples,
//! closures, channels, handles) and `extern "C"` functions that are called
//! by Cranelift-generated machine code.
//!
//! # Value Representation
//!
//! The compiler uses static types, so most values are unboxed:
//! - `i64` for Int, `f64` for Float, `i8` for Bool, zero-size for Nil
//! - Pointer types (`*mut TokString`, etc.) for heap objects
//! - `TokValue` tagged union (16 bytes) for the `Any` type
//!
//! # Reference Counting
//!
//! All heap types use `AtomicU32` for the refcount. Allocated via
//! `Box::into_raw`, freed via `Box::from_raw` when refcount reaches 0.

pub mod value;
pub mod string;
pub mod array;
pub mod map;
pub mod tuple;
pub mod closure;
pub mod channel;
pub mod handle;
pub mod builtins;

// Re-export core types for convenience
pub use value::{TokValue, TokValueData};
pub use string::TokString;
pub use array::TokArray;
pub use map::TokMap;
pub use tuple::TokTuple;
pub use closure::TokClosure;
pub use channel::TokChannel;
pub use handle::TokHandle;

// Re-export tag constants
pub use value::{
    TAG_NIL, TAG_INT, TAG_FLOAT, TAG_BOOL, TAG_STRING,
    TAG_ARRAY, TAG_MAP, TAG_TUPLE, TAG_FUNC, TAG_CHANNEL, TAG_HANDLE,
};
