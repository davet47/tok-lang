# Tok Language — Tech Debt Tracker

Last updated: 2025-06-20 — 10 open items (79 fixed of 89 total)

---

## LOW

### 43. Magic numbers without constants (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs`
`(i * 16)` for TokValue offset, tag/data field sizes, etc. Should define `const TOKVALUE_SIZE: usize = 16`.

### 44. String index overflow (tok-runtime)
**File:** `crates/tok-runtime/src/strings.rs`
String indexing doesn't check for out-of-bounds access consistently. Some paths return empty string, others may panic.

### 45. Dead code markers (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (lines ~157, 168, 684, 693)
`#[allow(dead_code)]` on fields like `call_conv`, `gensym_counter`, `return_var`, `ret_type`. Remove if truly unused, document if kept for future use.

### 46. Undocumented ordering in select (tok-runtime)
**File:** `crates/tok-runtime/src/channel.rs`
Select tries arms in order (not random like Go). This semantic difference isn't documented and could surprise users expecting Go-like behavior.

### 47. Incomplete pattern checking in compile_binop (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (line ~2881)
Bitwise op fallback silently returns 0 instead of panicking. Should `panic!("Unhandled binop")` for defense.

### 48. Negative index wrapping through unsigned cast (tok-runtime)
**Files:** `crates/tok-runtime/src/string.rs` (lines 128-132)
Negative index calculation `(len as i64 + i) as usize` can wrap to a huge `usize` when `|i| > len`. Subsequent bounds checks catch it, but the pattern is fragile. Should validate as `i64` before casting.
*Note: tuple.rs was fixed in #34 (medium). string.rs still has this pattern.*

### 49. `tok_math_abs_t` returns 0 for non-numeric input (tok-runtime)
**File:** `crates/tok-runtime/src/stdlib_math.rs` (line 99)
Returns `TokValue::from_int(0)` for strings, arrays, etc. instead of Nil. Silently masking type errors.

### 50. `tok_http_serve_t` blocks forever with no shutdown mechanism (tok-runtime)
**File:** `crates/tok-runtime/src/stdlib_http.rs` (lines 308-486)
Loops forever on `listener.incoming()` with no way to stop from Tok code. Should document limitation or provide a shutdown channel.

### 51. `HashableTokValue` uses magic number `3` instead of `TAG_BOOL` (tok-runtime)
**File:** `crates/tok-runtime/src/array.rs` (lines 29, 551)
`3 => v.data.bool_val.hash(state)` — should use `TAG_BOOL` constant for consistency.

### 52. LLM stdlib uses manual `format!` JSON construction (tok-runtime)
**File:** `crates/tok-runtime/src/stdlib_llm.rs`
HTTP request bodies built with `format!()` string interpolation. Fragile for prompts containing quotes/backslashes. Crate already depends on `serde_json`.
**Fix:** Use `serde_json::json!()` or `serde_json::to_string()`.

---

## Completed (79 items)

### Previously completed (46 items — before 2025-06-20)

- [x] #1 COW string race condition — `Acquire` ordering on COW check + `Acquire` fence in `rc_dec` — *2025-06-14*
- [x] #2 `arg_to_str` lifetime — extracted `stdlib_helpers.rs`, deduplicated 18 definitions — *2025-06-14*
- [x] #3 Unsafe pointer aliasing in codegen — removed 4 raw pointer casts, NLL handles it — *2025-06-14*
- [x] #4 Import handler duplication — extracted `ImportCtx::load_module()` — *2025-06-14*
- [x] #5 O(n²) uniq — `HashSet<HashableTokValue>` for O(n) dedup — *2025-06-14*
- [x] #6 Bare `unwrap()` in codegen — 89 replaced with contextual `expect()` — *2025-06-14*
- [x] #7 Bare `unwrap()` in channel ops — `unwrap_or_else(|e| e.into_inner())` — *2025-06-14*
- [x] #8 Unsafe transmute — `TokValue::from_tag_data()` — *2025-06-14*
- [x] #9 Lexer duplication — extracted `collect_text`, `consume_exponent`, `lex_radix_int` — *2025-06-14*
- [x] #10 Type inference duplication — unified into `tok-types` free functions — *2025-06-14*
- [x] #11 Variadic flag lost in HIR — added `variadic: bool` to `HirParam` — *2025-06-14*
- [x] #12 Unbounded recursion in HIR visitor — depth limits at 1000 — *2025-06-14*
- [x] #13 O(n²) rename loop — `BatchRenamer` with HashMap — *2025-06-14*
- [x] #14 Thread-unsafe PRNG — `thread_local! { Cell<u64> }` — *2025-06-14*
- [x] #15 Unbounded recursion in `flatten_into` — `MAX_FLATTEN_DEPTH = 1000` — *2025-06-14*
- [x] #16 Poisoned-mutex in handle.rs — `unwrap_or_else` — *2025-06-14*
- [x] #17 `insert_func` duplicated 13x — canonical version in `stdlib_helpers.rs` — *2025-06-14*
- [x] #18 `i64::abs()` overflow — `.saturating_abs()` — *2025-06-14*
- [x] #19 Bare `unwrap()` on `to_str()` — `.to_string_lossy()` — *2025-06-14*
- [x] #20 String length byte vs char — `.chars().count()` — *2025-06-14*
- [x] #21 SPSC ring buffer in MPMC context — replaced with `Mutex<VecDeque>` bounded queue — *2025-06-19*
- [x] #22 pmap shared env_ptr — deep-copy env per thread, added `env_count` to `TokClosure` — *2025-06-19*
- [x] #23 `int()` trapping `fcvt_to_sint` — changed to `fcvt_to_sint_sat` — *2025-06-19*
- [x] #24 Specialized lambda return var wrong type — use `zero_value()` helper — *2025-06-19*
- [x] #25 IndexAssign/MemberAssign no-op on Any — added `tok_value_index_set` runtime + map ptr extraction — *2025-06-19*
- [x] #26 f64→i64 cast UB — `safe_f64_to_i64()` helper clamps NaN/Inf/out-of-range — *2025-06-19*
- [x] #27 `tok_value_negate` i64::MIN panic — `.wrapping_neg()` — *2025-06-19*
- [x] #28 Integer overflow in add/sub/mul — `wrapping_add`/`wrapping_sub`/`wrapping_mul` — *2025-06-19*
- [x] #29 Member access key string leaks — `tok_string_free` after map ops in codegen — *2025-06-19*
- [x] #30 Tuple member allocates unused key string — moved alloc below Tuple arm — *2025-06-19*
- [x] #31 assert! in extern C is UB — `null_check!` macro with `abort()` across 67 sites — *2025-06-19*
- [x] #32 Lexer interpolation missing operators — added ~15 operators to sub-lexer — *2025-06-19*
- [x] #33 str.index_of byte vs char offset — `s[..byte_pos].chars().count()` — *2025-06-19*
- [x] #34 HTTP serve leaks request maps — `rc_dec` after handler returns — *2025-06-19*
- [x] #35 Pattern::Guard placeholder — `eprintln!` warning for unreachable path — *2025-06-19*
- [x] #36 While/Infinite loops jump after terminated body — `block_terminated` guard — *2025-06-19*
- [x] #37 `sdiv`/`srem` trap on div-by-zero and `i64::MIN / -1` — safe branching in codegen — *2025-06-19*
- [x] #38 `i64::MIN / -1` overflow in runtime div/mod — wrapping guard in `tok_value_div`/`tok_value_mod` — *2025-06-19*
- [x] #39 `tok_pmap` elements not `rc_inc`'d before thread dispatch — added `rc_inc` per element — *2025-06-19*
- [x] #40 HTTP serve handler result not freed — `result.rc_dec()` after extracting response — *2025-06-19*
- [x] #41 LLM `tok_llm_ask_t` temporaries leaked — `rc_dec` msg array + empty opts — *2025-06-19*
- [x] #42 Non-string map index key leaked — `tok_string_free` after `tok_map_set` — *2025-06-19*
- [x] #43 Closure `env_ptr` not freed on TAG_FUNC rc_dec — `tok_env_free` before drop — *2025-06-19*
- [x] #44 HTTP serve handler ABI mismatch — returns `(i64, i64)` not `TokValue` — *2025-06-19*
- [x] #45 Inconsistent error handling — `DriverError` enum, documented convention per crate — *2025-06-19*
- [x] #46 Missing rc_inc for captured heap values + Any array push crash — `tok_value_rc_inc` in closure/goroutine envs, `unwrap_any_ptr` in tok_array_push RuntimeCall — *2025-06-19*

### Medium-priority items completed (33 items — 2025-06-20)

- [x] M#10 `compile_stmt` too long — extracted `compile_stmt` into focused helper functions — *2025-06-20*
- [x] M#11 `compile_expr` too large — split into focused helper functions — *2025-06-20*
- [x] M#12 `compile_call` massive match — split builtin match into category helpers — *2025-06-20*
- [x] M#13 Type coercion scattered — documented type coercion system conventions — *2025-06-20*
- [x] M#14 Loop duplication: unrolled vs non-unrolled — extracted `emit_loop_increment` helper — *2025-06-20*
- [x] M#15 Capture loading repeated — extracted `load_captures_from_env` helper — *2025-06-20*
- [x] M#16 Parser statement parsing duplication — added `LookaheadCursor` to unify lookahead patterns — *2025-06-20*
- [x] M#17 Stdlib module duplication — added `math_f64_unary!`, `math_f64_to_int!`, `math_f64_binary!`, `str_unary!` macros — *2025-06-20*
- [x] M#19 Hardcoded stdlib list — single source of truth via `STDLIB_MODULE_CONSTRUCTORS` and `is_stdlib_module()` — *2025-06-20*
- [x] M#20 `compile_expr` Option<Value> unclear — added comprehensive doc comment explaining semantics — *2025-06-20*
- [x] M#21 Missing `compile_and_unwrap_ptr` helper — added `compile_expr_as_ptr` combining compile + unwrap — *2025-06-20*
- [x] M#22 Regex recompilation on every call — thread-local `HashMap<String, Regex>` cache (capacity 64) — *2025-06-20*
- [x] M#23 `tok_string_index` allocates Vec — changed to `chars().nth()` — *2025-06-20*
- [x] M#24 `tok_array_sort`/`tok_array_rev` double-clone — `rc_inc` before clone, then sort/reverse in place — *2025-06-20*
- [x] M#25 `tok_map_del` copies entire map — clone-then-remove instead of rebuild — *2025-06-20*
- [x] M#26 `tok_env_alloc`/`tok_env_free` panic on Layout failure — match with error handling — *2025-06-20*
- [x] M#27 Blanket clippy allow — added doc comment explaining rationale for crate-wide scope — *2025-06-20*
- [x] M#28 `tok_array_sum` integer overflow — `checked_add` with float fallback — *2025-06-20*
- [x] M#29 Codegen `to_bool` truthiness divergence — delegates to runtime `tok_value_truthiness` for Str/Array/Map — *2025-06-20*
- [x] M#30 `retype_expr` incomplete recursion — added recursion into Index, Member, If, Block, Array, Tuple, Length — *2025-06-20*
- [x] M#31 `is_heap_type` missing Optional/Result — added `Type::Optional(_) | Type::Result(_)` — *2025-06-20*
- [x] M#32 TCO tail-call skips Nil-typed args — push zero values to match block params — *2025-06-20*
- [x] M#33 `chan()`/`exit()` don't unwrap Any-typed args — added `from_tokvalue_raw_data` helper for payload extraction — *2025-06-20*
- [x] M#34 `tok_tuple_get` negative indices — wrap from end, matching array behavior — *2025-06-20*
- [x] M#35 Unbounded recursion in JSON/TOON/template parsers — depth limit at 128 — *2025-06-20*
- [x] M#36 `tok_string_repeat` unbounded allocation — capped at 1M characters — *2025-06-20*
- [x] M#37 `io.input` registered twice — removed dead 0-arg registration — *2025-06-20*
- [x] M#38 Duplicate `to_result_tuple` — moved to `stdlib_helpers.rs` — *2025-06-20*
- [x] M#39 Duplicate PRNG implementations — unified into shared `xorshift_rand()` in `stdlib_helpers` — *2025-06-20*
- [x] M#40 `try_member_index_assign` backtrack re-parse — replaced with chain-walking lookahead — *2025-06-20*
- [x] M#41 `check_block_stmts` double-checks last expr — capture type directly from `check_expr` — *2025-06-20*
- [x] M#42 `find_runtime_lib` relative paths — resolve relative to executable, not CWD — *2025-06-20*
