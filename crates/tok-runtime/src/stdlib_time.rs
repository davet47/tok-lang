//! Standard library: `@"time"` module.
//!
//! Provides time functions: now, sleep, fmt.

use crate::closure::TokClosure;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{TokValue, TAG_FLOAT, TAG_INT};

use std::time::{SystemTime, UNIX_EPOCH};

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

#[inline]
fn arg_to_f64(tag: i64, data: i64) -> f64 {
    match tag as u8 {
        TAG_FLOAT => f64::from_bits(data as u64),
        TAG_INT => data as f64,
        _ => 0.0,
    }
}

#[inline]
fn arg_to_i64(tag: i64, data: i64) -> i64 {
    match tag as u8 {
        TAG_INT => data,
        TAG_FLOAT => f64::from_bits(data as u64) as i64,
        _ => 0,
    }
}

#[inline]
unsafe fn arg_to_str<'a>(tag: i64, data: i64) -> &'a str {
    if tag as u8 == crate::value::TAG_STRING {
        let ptr = data as *mut TokString;
        if !ptr.is_null() {
            return &(*ptr).data;
        }
    }
    ""
}

// ═══════════════════════════════════════════════════════════════
// Trampolines
// ═══════════════════════════════════════════════════════════════

/// now() -> Float (unix timestamp in seconds with sub-second precision)
#[no_mangle]
pub extern "C" fn tok_time_now_t(_env: *mut u8) -> TokValue {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    TokValue::from_float(dur.as_secs_f64())
}

/// sleep(ms) -> Nil (sleep for milliseconds)
#[no_mangle]
pub extern "C" fn tok_time_sleep_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    let ms = arg_to_i64(tag, data);
    if ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
    }
    TokValue::nil()
}

/// fmt(timestamp, pattern) -> Str
/// Formats a unix timestamp using a simple pattern.
/// Supported: %Y (year), %m (month), %d (day), %H (hour), %M (minute), %S (second)
#[no_mangle]
pub extern "C" fn tok_time_fmt_t(
    _env: *mut u8,
    tag1: i64, data1: i64,
    tag2: i64, data2: i64,
) -> TokValue {
    let ts = arg_to_f64(tag1, data1);
    let pattern = unsafe { arg_to_str(tag2, data2) };

    // Convert unix timestamp to broken-down time components
    let secs = ts as i64;
    let (year, month, day, hour, min, sec) = unix_to_datetime(secs);

    let result = pattern
        .replace("%Y", &format!("{:04}", year))
        .replace("%m", &format!("{:02}", month))
        .replace("%d", &format!("{:02}", day))
        .replace("%H", &format!("{:02}", hour))
        .replace("%M", &format!("{:02}", min))
        .replace("%S", &format!("{:02}", sec));

    TokValue::from_string(TokString::alloc(result))
}

/// Convert unix timestamp (seconds since epoch) to (year, month, day, hour, min, sec).
/// Simple implementation — handles dates from 1970 onward.
fn unix_to_datetime(secs: i64) -> (i64, i64, i64, i64, i64, i64) {
    let sec_in_min = 60i64;
    let sec_in_hour = 3600i64;
    let sec_in_day = 86400i64;

    let time_of_day = ((secs % sec_in_day) + sec_in_day) % sec_in_day;
    let hour = time_of_day / sec_in_hour;
    let min = (time_of_day % sec_in_hour) / sec_in_min;
    let sec = time_of_day % sec_in_min;

    let mut days = secs / sec_in_day;
    if secs < 0 && secs % sec_in_day != 0 {
        days -= 1;
    }

    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    days += 719468; // shift epoch from 1970-01-01 to 0000-03-01
    let era = if days >= 0 { days } else { days - 146096 } / 146097;
    let doe = days - era * 146097; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    (year, m, d, hour, min, sec)
}

// ═══════════════════════════════════════════════════════════════
// Module constructor
// ═══════════════════════════════════════════════════════════════

fn insert_func(m: *mut TokMap, name: &str, fn_ptr: *const u8, arity: u32) {
    let closure = TokClosure::alloc(fn_ptr, std::ptr::null_mut(), arity);
    let val = TokValue::from_func(closure);
    unsafe {
        (*m).data.insert(name.to_string(), val);
    }
}

#[no_mangle]
pub extern "C" fn tok_stdlib_time() -> *mut TokMap {
    let m = TokMap::alloc();

    insert_func(m, "now",   tok_time_now_t   as *const u8, 0);
    insert_func(m, "sleep", tok_time_sleep_t as *const u8, 1);
    insert_func(m, "fmt",   tok_time_fmt_t   as *const u8, 2);

    m
}
