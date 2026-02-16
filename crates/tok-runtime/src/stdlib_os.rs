//! Standard library: `@"os"` module.
//!
//! Provides OS-level functions: args, env, cwd, pid, exit, time, sleep.

use crate::array::TokArray;
use crate::closure::TokClosure;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{TokValue, TAG_FLOAT, TAG_INT, TAG_STRING};

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

#[inline]
unsafe fn arg_to_str<'a>(tag: i64, data: i64) -> &'a str {
    if tag as u8 == TAG_STRING {
        let ptr = data as *mut TokString;
        if !ptr.is_null() {
            return &(*ptr).data;
        }
    }
    ""
}

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
    if tag as u8 == TAG_INT { data } else { 0 }
}

// ═══════════════════════════════════════════════════════════════
// Trampolines
// ═══════════════════════════════════════════════════════════════

/// args() -> Array<Str>
#[no_mangle]
pub extern "C" fn tok_os_args_t(_env: *mut u8) -> TokValue {
    let arr = TokArray::alloc();
    for arg in std::env::args() {
        let s = TokString::alloc(arg);
        unsafe { (*arr).data.push(TokValue::from_string(s)); }
    }
    TokValue::from_array(arr)
}

/// env(name) -> Str | Nil
#[no_mangle]
pub extern "C" fn tok_os_env_t(_env_ptr: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let name = arg_to_str(tag, data);
        match std::env::var(name) {
            Ok(val) => TokValue::from_string(TokString::alloc(val)),
            Err(_) => TokValue::nil(),
        }
    }
}

/// set_env(name, value) -> Nil
#[no_mangle]
pub extern "C" fn tok_os_set_env_t(
    _env_ptr: *mut u8,
    tag1: i64, data1: i64,
    tag2: i64, data2: i64,
) -> TokValue {
    unsafe {
        let name = arg_to_str(tag1, data1).to_string();
        let val = arg_to_str(tag2, data2).to_string();
        std::env::set_var(&name, &val);
    }
    TokValue::nil()
}

/// cwd() -> Str
#[no_mangle]
pub extern "C" fn tok_os_cwd_t(_env: *mut u8) -> TokValue {
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    TokValue::from_string(TokString::alloc(cwd))
}

/// pid() -> Int
#[no_mangle]
pub extern "C" fn tok_os_pid_t(_env: *mut u8) -> TokValue {
    TokValue::from_int(std::process::id() as i64)
}

/// exit(code) -> !
#[no_mangle]
pub extern "C" fn tok_os_exit_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    let code = arg_to_i64(tag, data) as i32;
    std::process::exit(code);
}

/// time() -> Float (unix timestamp in seconds)
#[no_mangle]
pub extern "C" fn tok_os_time_t(_env: *mut u8) -> TokValue {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    TokValue::from_float(dur.as_secs_f64())
}

/// sleep(seconds) -> Nil
#[no_mangle]
pub extern "C" fn tok_os_sleep_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    let secs = arg_to_f64(tag, data);
    if secs > 0.0 {
        std::thread::sleep(std::time::Duration::from_secs_f64(secs));
    }
    TokValue::nil()
}

/// exec(cmd) -> (stdout, exit_code)
#[no_mangle]
pub extern "C" fn tok_os_exec_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let cmd = arg_to_str(tag, data).to_string();
        match std::process::Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .output()
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let code = output.status.code().unwrap_or(-1) as i64;
                let elems = vec![
                    TokValue::from_string(TokString::alloc(stdout)),
                    TokValue::from_int(code),
                ];
                TokValue::from_tuple(crate::tuple::TokTuple::alloc(elems))
            }
            Err(e) => {
                let elems = vec![
                    TokValue::nil(),
                    TokValue::from_string(TokString::alloc(e.to_string())),
                ];
                TokValue::from_tuple(crate::tuple::TokTuple::alloc(elems))
            }
        }
    }
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
pub extern "C" fn tok_stdlib_os() -> *mut TokMap {
    let m = TokMap::alloc();

    // 0-arg functions
    insert_func(m, "args",  tok_os_args_t  as *const u8, 0);
    insert_func(m, "cwd",   tok_os_cwd_t   as *const u8, 0);
    insert_func(m, "pid",   tok_os_pid_t   as *const u8, 0);
    insert_func(m, "time",  tok_os_time_t  as *const u8, 0);

    // 1-arg functions
    insert_func(m, "env",   tok_os_env_t   as *const u8, 1);
    insert_func(m, "exit",  tok_os_exit_t  as *const u8, 1);
    insert_func(m, "sleep", tok_os_sleep_t as *const u8, 1);
    insert_func(m, "exec",  tok_os_exec_t  as *const u8, 1);

    // 2-arg functions
    insert_func(m, "set_env", tok_os_set_env_t as *const u8, 2);

    m
}
