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

// Crate-wide allows: nearly every `extern "C" fn` in this crate takes raw
// pointers and dereferences them after null_check!. Annotating each function
// individually would add noise without improving safety. The null_check! macro
// already aborts on null pointers before any dereference.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(dangerous_implicit_autorefs)]

/// Null-pointer check for `extern "C"` functions. Calls `abort()` instead
/// of `panic!()` to avoid unwinding across the FFI boundary (which is UB).
#[macro_export]
macro_rules! null_check {
    ($ptr:expr, $msg:literal) => {
        if $ptr.is_null() {
            eprintln!("fatal: {}", $msg);
            std::process::abort();
        }
    };
}

pub mod array;
pub mod builtins;
pub mod channel;
pub mod closure;
pub mod handle;
pub mod map;
pub mod stdlib_csv;
pub mod stdlib_fs;
pub mod stdlib_helpers;
pub mod stdlib_http;
pub mod stdlib_io;
pub mod stdlib_json;
pub mod stdlib_llm;
pub mod stdlib_math;
pub mod stdlib_os;
pub mod stdlib_re;
pub mod stdlib_str;
pub mod stdlib_time;
pub mod stdlib_tmpl;
pub mod stdlib_toon;
pub mod string;
pub mod tuple;
pub mod value;

// Re-export core types for convenience
pub use array::TokArray;
pub use channel::TokChannel;
pub use closure::TokClosure;
pub use handle::TokHandle;
pub use map::TokMap;
pub use string::TokString;
pub use tuple::TokTuple;
pub use value::{TokValue, TokValueData};

// Re-export tag constants
pub use value::{
    TAG_ARRAY, TAG_BOOL, TAG_CHANNEL, TAG_FLOAT, TAG_FUNC, TAG_HANDLE, TAG_INT, TAG_MAP, TAG_NIL,
    TAG_STRING, TAG_TUPLE,
};
