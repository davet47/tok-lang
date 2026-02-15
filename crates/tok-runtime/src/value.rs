//! Tagged union `TokValue` for the `a` (Any) type.
//!
//! Layout: 16 bytes — 8-byte tag (padded) + 8-byte payload.
//! All pointer payloads are `*mut T` to refcounted heap objects.

use std::fmt;

use crate::string::TokString;
use crate::array::TokArray;
use crate::map::TokMap;
use crate::tuple::TokTuple;
use crate::channel::TokChannel;
use crate::handle::TokHandle;
use crate::closure::TokClosure;

// ═══════════════════════════════════════════════════════════════
// Tag constants
// ═══════════════════════════════════════════════════════════════

pub const TAG_NIL: u8 = 0;
pub const TAG_INT: u8 = 1;
pub const TAG_FLOAT: u8 = 2;
pub const TAG_BOOL: u8 = 3;
pub const TAG_STRING: u8 = 4;
pub const TAG_ARRAY: u8 = 5;
pub const TAG_MAP: u8 = 6;
pub const TAG_TUPLE: u8 = 7;
pub const TAG_FUNC: u8 = 8;
pub const TAG_CHANNEL: u8 = 9;
pub const TAG_HANDLE: u8 = 10;

// ═══════════════════════════════════════════════════════════════
// TokValue — tagged union
// ═══════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Copy, Clone)]
pub union TokValueData {
    pub int_val: i64,
    pub float_val: f64,
    pub bool_val: i8,
    pub string_ptr: *mut TokString,
    pub array_ptr: *mut TokArray,
    pub map_ptr: *mut TokMap,
    pub tuple_ptr: *mut TokTuple,
    pub func_ptr: *mut TokClosure,
    pub channel_ptr: *mut TokChannel,
    pub handle_ptr: *mut TokHandle,
    pub _raw: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TokValue {
    pub tag: u8,
    _pad: [u8; 7],
    pub data: TokValueData,
}

// Safety: TokValue is a plain data type with no thread-local state.
// Pointer payloads point to Arc-like refcounted objects that are Send+Sync.
unsafe impl Send for TokValue {}
unsafe impl Sync for TokValue {}

impl TokValue {
    pub fn nil() -> Self {
        TokValue {
            tag: TAG_NIL,
            _pad: [0; 7],
            data: TokValueData { _raw: 0 },
        }
    }

    pub fn from_int(v: i64) -> Self {
        TokValue {
            tag: TAG_INT,
            _pad: [0; 7],
            data: TokValueData { int_val: v },
        }
    }

    pub fn from_float(v: f64) -> Self {
        TokValue {
            tag: TAG_FLOAT,
            _pad: [0; 7],
            data: TokValueData { float_val: v },
        }
    }

    pub fn from_bool(v: bool) -> Self {
        TokValue {
            tag: TAG_BOOL,
            _pad: [0; 7],
            data: TokValueData {
                bool_val: if v { 1 } else { 0 },
            },
        }
    }

    pub fn from_string(ptr: *mut TokString) -> Self {
        TokValue {
            tag: TAG_STRING,
            _pad: [0; 7],
            data: TokValueData { string_ptr: ptr },
        }
    }

    pub fn from_array(ptr: *mut TokArray) -> Self {
        TokValue {
            tag: TAG_ARRAY,
            _pad: [0; 7],
            data: TokValueData { array_ptr: ptr },
        }
    }

    pub fn from_map(ptr: *mut TokMap) -> Self {
        TokValue {
            tag: TAG_MAP,
            _pad: [0; 7],
            data: TokValueData { map_ptr: ptr },
        }
    }

    pub fn from_tuple(ptr: *mut TokTuple) -> Self {
        TokValue {
            tag: TAG_TUPLE,
            _pad: [0; 7],
            data: TokValueData { tuple_ptr: ptr },
        }
    }

    pub fn from_func(ptr: *mut TokClosure) -> Self {
        TokValue {
            tag: TAG_FUNC,
            _pad: [0; 7],
            data: TokValueData { func_ptr: ptr },
        }
    }

    pub fn from_channel(ptr: *mut TokChannel) -> Self {
        TokValue {
            tag: TAG_CHANNEL,
            _pad: [0; 7],
            data: TokValueData { channel_ptr: ptr },
        }
    }

    pub fn from_handle(ptr: *mut TokHandle) -> Self {
        TokValue {
            tag: TAG_HANDLE,
            _pad: [0; 7],
            data: TokValueData { handle_ptr: ptr },
        }
    }

    /// Increment reference count if this value holds a pointer type.
    pub fn rc_inc(&self) {
        unsafe {
            match self.tag {
                TAG_STRING => {
                    let p = self.data.string_ptr;
                    if !p.is_null() {
                        (*p).rc_inc();
                    }
                }
                TAG_ARRAY => {
                    let p = self.data.array_ptr;
                    if !p.is_null() {
                        (*p).rc_inc();
                    }
                }
                TAG_MAP => {
                    let p = self.data.map_ptr;
                    if !p.is_null() {
                        (*p).rc_inc();
                    }
                }
                TAG_TUPLE => {
                    let p = self.data.tuple_ptr;
                    if !p.is_null() {
                        (*p).rc_inc();
                    }
                }
                TAG_FUNC => {
                    let p = self.data.func_ptr;
                    if !p.is_null() {
                        (*p).rc_inc();
                    }
                }
                TAG_CHANNEL => {
                    let p = self.data.channel_ptr;
                    if !p.is_null() {
                        (*p).rc_inc();
                    }
                }
                TAG_HANDLE => {
                    let p = self.data.handle_ptr;
                    if !p.is_null() {
                        (*p).rc_inc();
                    }
                }
                _ => {}
            }
        }
    }

    /// Decrement reference count if this value holds a pointer type.
    /// Frees the object when the count reaches zero.
    pub fn rc_dec(&self) {
        unsafe {
            match self.tag {
                TAG_STRING => {
                    let p = self.data.string_ptr;
                    if !p.is_null() && (*p).rc_dec() {
                        drop(Box::from_raw(p));
                    }
                }
                TAG_ARRAY => {
                    let p = self.data.array_ptr;
                    if !p.is_null() && (*p).rc_dec() {
                        // Dec all elements before dropping
                        for v in &(*p).data {
                            v.rc_dec();
                        }
                        drop(Box::from_raw(p));
                    }
                }
                TAG_MAP => {
                    let p = self.data.map_ptr;
                    if !p.is_null() && (*p).rc_dec() {
                        for (_, v) in &(*p).data {
                            v.rc_dec();
                        }
                        drop(Box::from_raw(p));
                    }
                }
                TAG_TUPLE => {
                    let p = self.data.tuple_ptr;
                    if !p.is_null() && (*p).rc_dec() {
                        for v in &(*p).data {
                            v.rc_dec();
                        }
                        drop(Box::from_raw(p));
                    }
                }
                TAG_FUNC => {
                    let p = self.data.func_ptr;
                    if !p.is_null() && (*p).rc_dec() {
                        drop(Box::from_raw(p));
                    }
                }
                TAG_CHANNEL => {
                    let p = self.data.channel_ptr;
                    if !p.is_null() && (*p).rc_dec() {
                        drop(Box::from_raw(p));
                    }
                }
                TAG_HANDLE => {
                    let p = self.data.handle_ptr;
                    if !p.is_null() && (*p).rc_dec() {
                        drop(Box::from_raw(p));
                    }
                }
                _ => {}
            }
        }
    }

    /// Returns true if this value is truthy (Tok semantics).
    pub fn truthiness(&self) -> bool {
        unsafe {
            match self.tag {
                TAG_NIL => false,
                TAG_BOOL => self.data.bool_val != 0,
                TAG_INT => self.data.int_val != 0,
                TAG_FLOAT => self.data.float_val != 0.0,
                TAG_STRING => {
                    let p = self.data.string_ptr;
                    !p.is_null() && !(*p).data.is_empty()
                }
                TAG_ARRAY => {
                    let p = self.data.array_ptr;
                    !p.is_null() && !(*p).data.is_empty()
                }
                TAG_MAP => {
                    let p = self.data.map_ptr;
                    !p.is_null() && !(*p).data.is_empty()
                }
                // Tuples, funcs, channels, handles are always truthy
                _ => true,
            }
        }
    }

    /// Type name as a string slice.
    pub fn type_name(&self) -> &'static str {
        match self.tag {
            TAG_NIL => "nil",
            TAG_INT => "int",
            TAG_FLOAT => "float",
            TAG_BOOL => "bool",
            TAG_STRING => "string",
            TAG_ARRAY => "array",
            TAG_MAP => "map",
            TAG_TUPLE => "tuple",
            TAG_FUNC => "function",
            TAG_CHANNEL => "channel",
            TAG_HANDLE => "handle",
            _ => "unknown",
        }
    }
}

/// Format a float the Tok way: trim unnecessary trailing zeros,
/// but always keep at least one decimal place.
/// `3.0` stays as `3.0`, `3.10` becomes `3.1`, `3.00` becomes `3.0`.
pub fn format_float(f: f64) -> String {
    if f.is_infinite() {
        if f.is_sign_positive() {
            return "Inf".to_string();
        } else {
            return "-Inf".to_string();
        }
    }
    if f.is_nan() {
        return "NaN".to_string();
    }
    // Use enough precision to be lossless, then trim trailing zeros
    // but always keep at least one decimal digit.
    let s = format!("{:.}", f);
    if s.contains('.') {
        // Trim trailing zeros after the decimal point, but keep at least one
        let trimmed = s.trim_end_matches('0');
        if trimmed.ends_with('.') {
            format!("{}0", trimmed)
        } else {
            trimmed.to_string()
        }
    } else {
        format!("{}.0", s)
    }
}

impl fmt::Display for TokValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            match self.tag {
                TAG_NIL => write!(f, "N"),
                TAG_INT => write!(f, "{}", self.data.int_val),
                TAG_FLOAT => write!(f, "{}", format_float(self.data.float_val)),
                TAG_BOOL => {
                    if self.data.bool_val != 0 {
                        write!(f, "T")
                    } else {
                        write!(f, "F")
                    }
                }
                TAG_STRING => {
                    let p = self.data.string_ptr;
                    if p.is_null() {
                        write!(f, "N")
                    } else {
                        write!(f, "{}", (*p).data)
                    }
                }
                TAG_ARRAY => {
                    let p = self.data.array_ptr;
                    if p.is_null() {
                        write!(f, "[]")
                    } else {
                        write!(f, "[")?;
                        for (i, v) in (*p).data.iter().enumerate() {
                            if i > 0 {
                                write!(f, " ")?;
                            }
                            write!(f, "{}", v)?;
                        }
                        write!(f, "]")
                    }
                }
                TAG_MAP => {
                    let p = self.data.map_ptr;
                    if p.is_null() {
                        write!(f, "{{}}")
                    } else {
                        write!(f, "{{")?;
                        for (i, (k, v)) in (*p).data.iter().enumerate() {
                            if i > 0 {
                                write!(f, " ")?;
                            }
                            write!(f, "{}:{}", k, v)?;
                        }
                        write!(f, "}}")
                    }
                }
                TAG_TUPLE => {
                    let p = self.data.tuple_ptr;
                    if p.is_null() {
                        write!(f, "()")
                    } else {
                        write!(f, "(")?;
                        for (i, v) in (*p).data.iter().enumerate() {
                            if i > 0 {
                                write!(f, " ")?;
                            }
                            write!(f, "{}", v)?;
                        }
                        write!(f, ")")
                    }
                }
                TAG_FUNC => write!(f, "<function>"),
                TAG_CHANNEL => write!(f, "<channel>"),
                TAG_HANDLE => write!(f, "<handle>"),
                _ => write!(f, "<unknown>"),
            }
        }
    }
}

impl fmt::Debug for TokValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TokValue(tag={}, {})", self.tag, self)
    }
}

impl Default for TokValue {
    fn default() -> Self {
        Self::nil()
    }
}

impl PartialEq for TokValue {
    fn eq(&self, other: &Self) -> bool {
        if self.tag != other.tag {
            return false;
        }
        unsafe {
            match self.tag {
                TAG_NIL => true,
                TAG_INT => self.data.int_val == other.data.int_val,
                TAG_FLOAT => self.data.float_val == other.data.float_val,
                TAG_BOOL => self.data.bool_val == other.data.bool_val,
                TAG_STRING => {
                    let a = self.data.string_ptr;
                    let b = other.data.string_ptr;
                    if a.is_null() && b.is_null() {
                        return true;
                    }
                    if a.is_null() || b.is_null() {
                        return false;
                    }
                    (*a).data == (*b).data
                }
                // For pointer types, compare by pointer identity
                _ => self.data._raw == other.data._raw,
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nil() {
        let v = TokValue::nil();
        assert_eq!(v.tag, TAG_NIL);
        assert_eq!(format!("{}", v), "N");
        assert!(!v.truthiness());
    }

    #[test]
    fn test_int() {
        let v = TokValue::from_int(42);
        assert_eq!(v.tag, TAG_INT);
        assert_eq!(format!("{}", v), "42");
        assert!(v.truthiness());

        let z = TokValue::from_int(0);
        assert!(!z.truthiness());
    }

    #[test]
    fn test_float() {
        let v = TokValue::from_float(3.14);
        assert_eq!(v.tag, TAG_FLOAT);
        // format_float produces "3.14"
        assert_eq!(format!("{}", v), "3.14");
        assert!(v.truthiness());

        let z = TokValue::from_float(0.0);
        assert!(!z.truthiness());
        assert_eq!(format!("{}", z), "0.0");
    }

    #[test]
    fn test_bool() {
        let t = TokValue::from_bool(true);
        assert_eq!(format!("{}", t), "T");
        assert!(t.truthiness());

        let f = TokValue::from_bool(false);
        assert_eq!(format!("{}", f), "F");
        assert!(!f.truthiness());
    }

    #[test]
    fn test_type_names() {
        assert_eq!(TokValue::nil().type_name(), "nil");
        assert_eq!(TokValue::from_int(0).type_name(), "int");
        assert_eq!(TokValue::from_float(0.0).type_name(), "float");
        assert_eq!(TokValue::from_bool(true).type_name(), "bool");
    }

    #[test]
    fn test_equality() {
        assert_eq!(TokValue::from_int(42), TokValue::from_int(42));
        assert_ne!(TokValue::from_int(42), TokValue::from_int(43));
        assert_eq!(TokValue::nil(), TokValue::nil());
        assert_ne!(TokValue::from_int(0), TokValue::nil());
    }

    #[test]
    fn test_format_float() {
        assert_eq!(format_float(3.0), "3.0");
        assert_eq!(format_float(3.14), "3.14");
        assert_eq!(format_float(0.1), "0.1");
        assert_eq!(format_float(100.0), "100.0");
        assert_eq!(format_float(-1.5), "-1.5");
        assert_eq!(format_float(0.0), "0.0");
    }

    #[test]
    fn test_size() {
        assert_eq!(std::mem::size_of::<TokValue>(), 16);
    }
}

// ═══════════════════════════════════════════════════════════════
// Runtime helper: length for Any-typed values
// ═══════════════════════════════════════════════════════════════

/// Index into a TokValue (array, string, tuple).
/// Returns the element at the given index as a TokValue.
#[no_mangle]
pub extern "C" fn tok_value_index(val: TokValue, idx: i64) -> TokValue {
    unsafe {
        match val.tag {
            TAG_ARRAY => {
                let p = val.data.array_ptr;
                assert!(!p.is_null(), "tok_value_index: null array");
                crate::array::tok_array_get(p, idx)
            }
            TAG_STRING => {
                let p = val.data.string_ptr;
                assert!(!p.is_null(), "tok_value_index: null string");
                let result = crate::string::tok_string_index(p, idx);
                TokValue {
                    tag: TAG_STRING,
                    _pad: [0; 7],
                    data: TokValueData { string_ptr: result },
                }
            }
            TAG_TUPLE => {
                let p = val.data.tuple_ptr;
                assert!(!p.is_null(), "tok_value_index: null tuple");
                crate::tuple::tok_tuple_get(p, idx)
            }
            _ => TokValue::nil(),
        }
    }
}

/// Return the length of a TokValue. Works for arrays, strings, maps, tuples.
/// Returns 0 for non-collection types.
#[no_mangle]
pub extern "C" fn tok_value_len(val: TokValue) -> i64 {
    unsafe {
        match val.tag {
            TAG_STRING => {
                let p = val.data.string_ptr;
                if p.is_null() { 0 } else { (*p).data.len() as i64 }
            }
            TAG_ARRAY => {
                let p = val.data.array_ptr;
                if p.is_null() { 0 } else { (*p).data.len() as i64 }
            }
            TAG_MAP => {
                let p = val.data.map_ptr;
                if p.is_null() { 0 } else { (*p).data.len() as i64 }
            }
            TAG_TUPLE => {
                let p = val.data.tuple_ptr;
                if p.is_null() { 0 } else { (*p).data.len() as i64 }
            }
            _ => 0,
        }
    }
}
