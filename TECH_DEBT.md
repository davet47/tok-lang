# Tok Language — Tech Debt Tracker

Last updated: 2025-06-20 — 49 open items | 79 previously fixed

---

## HIGH

### 1. Unsafe `arg_to_str()` returns `'static` lifetime from non-static data (tok-runtime)
**File:** `crates/tok-runtime/src/stdlib_helpers.rs` (lines 40-52)
Returns `&'static str` from a `TokString` pointer whose lifetime is actually tied to the caller's stack frame. If a caller stores this reference beyond the current trampoline, it becomes a dangling reference.
**Fix:** Change return type to `&str` with a proper lifetime parameter, or wrap in a guard type.

### 2. `tok_pmap` memory leak on thread panic (tok-runtime)
**File:** `crates/tok-runtime/src/array.rs` (lines 555-557)
If a thread panics during `tok_pmap`, the join error path creates `TokValue::nil()` but the original element's refcount may not have been decremented. Repeated thread failures cause resource leaks.
**Fix:** Use RAII wrapper (struct that calls `rc_dec` in `Drop`) to guarantee cleanup in all paths.

### 3. `clone_env()` layout allocation panics on invalid size (tok-runtime)
**File:** `crates/tok-runtime/src/array.rs` (line 470)
`Layout::from_size_align(size, 8).unwrap()` will panic if `count * 16` overflows or produces an invalid layout. Same pattern in `free_env()` at line 496.
**Fix:** Use `checked_mul` for the size calculation and return null on layout error instead of panicking.

### 4. Excessive `clone()` on Token during parser pattern matching (tok-parser)
**File:** `crates/tok-parser/src/parser.rs` (lines 457, 774, 871, 1227, 1264, 1267, 1293, 1377, 1646, 1801)
`match self.peek().clone() { ... }` clones the entire token (including heap-allocated strings) when `peek()` returns `&Token`. 10+ occurrences, compounding on complex expressions.
**Fix:** Change to `match self.peek() { ... }` with reference patterns.

### 5. No bounds check on `-o` flag argument in driver (tok-driver)
**File:** `crates/tok-driver/src/main.rs` (lines 91-94)
`get_output_path()` accesses `args[i+1]` without checking bounds. Passing `-o` as the last argument causes a panic.
**Fix:** Check `i + 1 < args.len()` before indexing.

### 6. Missing validation of exported names in import resolver (tok-driver)
**File:** `crates/tok-driver/src/import_resolver.rs` (lines 129-160)
`extract_exports()` returns any public name without filtering reserved words or builtins. A user module can export `push`, `len`, etc., shadowing critical builtins.
**Fix:** Filter exports against a set of reserved/builtin names.

---

## MEDIUM

### 7. Integer overflow in `clone_env`/`free_env` size calculation (tok-runtime)
**File:** `crates/tok-runtime/src/array.rs` (lines 469, 490)
`let size = (count as usize) * 16;` can overflow silently on large count values, leading to undersized allocation and buffer overruns.
**Fix:** Use `(count as usize).checked_mul(16).expect("environment size overflow")`.

### 8. Missing null check on `fn_ptr` before transmute in closure calls (tok-runtime)
**File:** `crates/tok-runtime/src/array.rs` (lines 398, 431, 509-510)
`std::mem::transmute((*closure).fn_ptr)` is called without verifying `fn_ptr` is non-null. A null function pointer causes undefined behavior.
**Fix:** Add null check before transmute; abort on null.

### 9. Unsafe transmute in `tok_http_serve_t` without arity validation (tok-runtime)
**File:** `crates/tok-runtime/src/stdlib_http.rs` (line 390)
Transmutes `fn_ptr` to `extern "C" fn(...)` without verifying the closure's actual arity matches the expected signature.
**Fix:** Check `(*closure_ptr).arity` before calling.

### 10. `MAX_REPEAT_LEN` defined in two places (tok-runtime)
**File:** `crates/tok-runtime/src/string.rs` (line 141), `crates/tok-runtime/src/stdlib_str.rs` (line 152)
Same constant `1_000_000` defined independently in both files. Updating one without the other causes inconsistency.
**Fix:** Define once as `pub const` in `stdlib_helpers.rs`.

### 11. Closure env pointer cast without alignment validation (tok-runtime)
**File:** `crates/tok-runtime/src/array.rs` (lines 529-530, 535)
`thread_env_usize as *mut u8` reconstruction could create an unaligned pointer. Later `add(i * 16)` assumes proper alignment for TokValue access.
**Fix:** Ensure allocations use 16-byte alignment via `Layout`, or store alignment info.

### 12. `unreachable!()` in parser should be error returns (tok-parser)
**File:** `crates/tok-parser/src/parser.rs` (lines 431, 477, 670, 690, 704, 719, 1637, 1730, 1751, 1755, 1920)
11 calls to `unreachable!()` in paths that could theoretically be reached with malformed input. Panics instead of returning parse errors.
**Fix:** Replace with `Err(self.error(...))`.

### 13. Duplicate operator lexing in `lex_interpolation_expr()` (tok-lexer)
**File:** `crates/tok-lexer/src/lib.rs` (lines 912-1176)
~250 lines of token-matching logic duplicated from the main tokenization loop (lines 267-615). Operator changes must be updated in two places.
**Fix:** Extract operator tokenization into a shared helper.

### 14. Redundant UTF-8 validation in lexer (tok-lexer)
**File:** `crates/tok-lexer/src/lib.rs` (lines 593-610)
Manual UTF-8 byte sequence validation duplicates Rust's built-in handling. Input is `&str`, already guaranteed valid UTF-8.
**Fix:** Remove dead validation code.

### 15. Excessive `.clone()` in builtin type registration (tok-types)
**File:** `crates/tok-types/src/lib.rs` (lines 318-407)
`arr_any.clone()` and `map_any.clone()` called 20+ times during builtin registration. Creates unnecessary allocations at startup.
**Fix:** Pass references or use a macro to generate signatures without cloning.

### 16. Negative literal tuple index silently returns `Type::Any` (tok-types)
**File:** `crates/tok-types/src/lib.rs` (line 810)
`elts.get(*i as usize)` casts `i64` to `usize` without checking for negative values. Negative index wraps to huge positive, silently returning `Any` instead of an error.
**Fix:** Check `*i >= 0` before casting.

### 17. HIR lowerer re-infers types already computed by type checker (tok-hir)
**File:** `crates/tok-hir/src/lower.rs` (lines 103-274)
`infer_expr_type()` duplicates logic from `tok_types::check()`. The two can diverge, causing subtle type bugs.
**Fix:** Pass `TypeInfo` results to the lowerer instead of re-inferring.

### 18. `MAX_WALK_DEPTH` silently skips subtrees (tok-hir)
**File:** `crates/tok-hir/src/lower.rs` (line 182)
Hardcoded depth limit of 1000. When exceeded, the walker silently skips the subtree with only a warning — no error returned.
**Fix:** Return an error instead of silently skipping.

### 19. Circular import detection uses `Vec::contains()` — O(n) per check (tok-driver)
**File:** `crates/tok-driver/src/import_resolver.rs` (line 62)
`import_stack.contains(&canonical)` is O(n) linear search. Deeply nested imports degrade to O(n²).
**Fix:** Use `HashSet` for import_stack.

### 20. Fallback runtime lib search produces confusing linker error (tok-driver)
**File:** `crates/tok-driver/src/main.rs` (lines 272-274)
If runtime lib not found, falls back to bare `"libtok_runtime.a"`, producing an unhelpful linker "library not found" error.
**Fix:** Make this a hard error listing all searched paths.

### 21. Inefficient line-start detection in lexer (tok-lexer)
**File:** `crates/tok-lexer/src/lib.rs` (lines 215-234)
`is_at_line_start()` reverse-scans from current position to find newline — O(column_width) per call. Called on every `#` token.
**Fix:** Track line-start positions during tokenization; use direct lookup.

### 22. `parse_reduce_args()` ambiguous init vs function parsing (tok-parser)
**File:** `crates/tok-parser/src/parser.rs` (lines 1006-1021)
Cannot distinguish complex init expressions from the function argument in certain edge cases.
**Fix:** Use lookahead for lambda marker `\(` to reliably separate init from function.

### 23. Unsafe index access on `inst_results()`/`block_params()` (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (multiple: lines 1052-1053, 1091, 1292, 1334-1335, 1745, 1751, 1757, 2194-2195, etc.)
Direct `[0]`/`[1]` indexing on Cranelift instruction results without bounds checking. Mismatch between assumed and actual return count will panic.
**Fix:** Use `.get(0)` with proper error handling, or add debug assertions on result length.

### 24. Free variable collection clones `locals` set on every branch (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (lines 6092-6299)
`collect_free_vars_expr` clones the `locals` HashSet for if/else branches, loops, blocks. On deeply nested structures, this causes excessive allocation.
**Fix:** Use `Rc<HashSet>` or mutable references with scope management.

### 25. Missing error context in function declaration panics (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (lines 1599, 1611)
`unwrap_or_else(|| panic!(...))` doesn't include the function name in the error message.
**Fix:** Include function name in panic messages.

### 26. Null pointer risk in Any-type member access (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (lines 2272-2275)
Loads the data field at offset 8 from a TokValue without validating the pointer. If the TokValue is nil or corrupted, this is undefined behavior.
**Fix:** Add a runtime null/tag check before loading.

### 27. No cyclic detection for stdlib module self-imports (tok-driver)
**File:** `crates/tok-driver/src/import_resolver.rs` (line 100)
Stdlib modules bypass the file-import cycle detection path. A malformed stdlib module that imports itself would cause stack overflow.
**Fix:** Add cycle detection to stdlib loading path.

---

## LOW

### 28. Magic number `16` for TokValue size (tok-codegen, tok-runtime)
**Files:** `crates/tok-codegen/src/compiler.rs`, `crates/tok-runtime/src/array.rs` (lines 469, 479, 490, 496)
`(i * 16)` used for TokValue offset without a named constant. Not enforced by compile-time assertion.
**Fix:** Define `const TOKVALUE_SIZE: usize = 16` and add `const _: () = assert!(size_of::<TokValue>() == 16);`.

### 29. Dead code markers on compiler struct fields (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (lines ~157, 168, 207-208, 684, 693)
`#[allow(dead_code)]` on `call_conv`, `gensym_counter`, `return_var`, `ret_type`. Remove if truly unused, document if kept for future use.

### 30. Incomplete pattern checking in `compile_binop` (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (line ~2881)
Bitwise op fallback silently returns 0 instead of panicking. Should `panic!("Unhandled binop")` for defense.

### 31. String index overflow inconsistency (tok-runtime)
**File:** `crates/tok-runtime/src/string.rs` (lines 128-132)
Negative index calculation `(len as i64 + i) as usize` can wrap to huge `usize` when `|i| > len`. Subsequent bounds checks catch it, but the pattern is fragile.
**Fix:** Validate as `i64` before casting to `usize`.

### 32. `tok_math_abs_t` returns 0 for non-numeric input (tok-runtime)
**File:** `crates/tok-runtime/src/stdlib_math.rs` (line 99)
Returns `TokValue::from_int(0)` for strings, arrays, etc. instead of Nil. Silently masks type errors.

### 33. `tok_http_serve_t` blocks forever with no shutdown (tok-runtime)
**File:** `crates/tok-runtime/src/stdlib_http.rs` (lines 308-486)
Loops forever on `listener.incoming()` with no way to stop from Tok code.
**Fix:** Document limitation or provide a shutdown channel.

### 34. Undocumented select ordering (tok-runtime)
**File:** `crates/tok-runtime/src/channel.rs`
Select tries arms in order (not random like Go). This semantic difference could surprise users expecting Go-like behavior.
**Fix:** Document in language reference.

### 35. `HashableTokValue` uses magic number `3` instead of `TAG_BOOL` (tok-runtime)
**File:** `crates/tok-runtime/src/array.rs` (lines 29, 551)
`3 => v.data.bool_val.hash(state)` — should use `TAG_BOOL` constant.

### 36. LLM stdlib uses manual `format!` JSON construction (tok-runtime)
**File:** `crates/tok-runtime/src/stdlib_llm.rs`
HTTP request bodies built with `format!()` string interpolation. Fragile for prompts containing quotes/backslashes. Crate already depends on `serde_json`.
**Fix:** Use `serde_json::json!()` or `serde_json::to_string()`.

### 37. `Channel::try_recv()` conflates "no data" with "closed" (tok-runtime)
**File:** `crates/tok-runtime/src/channel.rs` (lines 219-250)
Returns `Option<TokValue>` but cannot distinguish between "channel empty" and "channel closed". Proper Go-like channels should signal closure separately.
**Fix:** Return a custom enum: `RecvResult { Ok(TokValue), Closed, Empty }`.

### 38. Closure signature cache not thread-safe (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (lines 762-763)
`closure_sig_cache: HashMap<usize, SigRef>` is a plain HashMap. Not a problem now (single-threaded codegen), but would be if parallelization is added.
**Fix:** Document single-threaded assumption; use concurrent map if needed later.

### 39. Hardcoded loop unroll factor (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (line 5157)
`UNROLL_FACTOR = 4` with no dynamic heuristic. Large loop bodies unrolled 4x could cause code bloat.
**Fix:** Add body-size heuristic or make configurable.

### 40. `expect()` in string data declaration has generic error message (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (lines 684, 689)
`expect()` calls will crash with generic messages if the linker fails to declare/define string data.
**Fix:** Return `Result<>` and propagate errors with context.

### 41. Unchecked `usize → i64` casts in codegen (tok-codegen)
**File:** `crates/tok-codegen/src/compiler.rs` (lines 1920, 1998, 2149, 2164, 2172, 2264, 2556, 2560, 2770)
Multiple `.len() as i64` casts without overflow checks. Unlikely with realistic inputs but fragile.
**Fix:** Use `TryInto` or add `debug_assert!` for overflow detection.

### 42. String allocation in `collect_text()` (tok-lexer)
**File:** `crates/tok-lexer/src/lib.rs` (lines 627-634)
Creates a new String for every number literal, filtering out underscores. Could filter in a single pass.

### 43. Incomplete `infer_expr_type()` for Match expressions (tok-hir)
**File:** `crates/tok-hir/src/lower.rs` (line 212)
`Expr::Match` returns `Type::Any` without walking match arms to unify their types (unlike `check_expr` in tok-types).
**Fix:** Implement arm type unification.

### 44. `TypeEnv::update()` defines in outermost scope on miss (tok-types)
**File:** `crates/tok-types/src/lib.rs` (lines 258-266)
If variable not found in any scope, `update()` falls back to `define()` in outermost scope via `last_mut()`. Could lead to unexpected scoping.
**Fix:** Clarify intent; add explicit scope target parameter.

### 45. No recursion depth limit on file import resolution (tok-driver)
**File:** `crates/tok-driver/src/import_resolver.rs` (lines 47-95)
`load_module()` recursively compiles imported files with no depth limit (beyond cycle detection).
**Fix:** Add recursion depth counter; error after N levels.

### 46. Thread join panics in tests lose context (tok-runtime)
**File:** `crates/tok-runtime/src/string.rs` (line 467), `crates/tok-runtime/src/channel.rs` (lines 365, 428, 459)
`.join().unwrap()` in tests swallows the original thread panic message.
**Fix:** Use `.join().expect("thread panicked: ...")`.

### 47. Parser backtracking invariant undocumented (tok-parser)
**File:** `crates/tok-parser/src/parser.rs` (lines 374, 410)
`try_member_index_assign()` backtracks on failure but the invariant that `pos == saved_pos` on `Ok(None)` is not asserted.
**Fix:** Add defensive assertion; document the invariant.

### 48. Redundant `at()` discriminant checks (tok-parser)
**File:** `crates/tok-parser/src/parser.rs` (lines 113-115)
`at()` uses `std::mem::discriminant()` where simple pattern matching would be clearer and potentially faster.

### 49. `cmd_run()` unused variable clarity (tok-driver)
**File:** `crates/tok-driver/src/main.rs` (line 231)
Extracts `s` from `Ok(s)` then uses `s.code()`. Slightly unclear intent.
**Fix:** Use `status.code().unwrap_or(1)` directly for clarity.

