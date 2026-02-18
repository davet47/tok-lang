# Tok Examples

Runnable examples demonstrating Tok language features. All examples compile to native binaries via the Cranelift codegen backend.

## Prerequisites

Build the Tok compiler from the repo root:

```bash
cargo build --release
```

This produces the `tok` binary at `./target/release/tok`.

## Running Examples

### Using the binary directly

```bash
./target/release/tok run examples/<file>.tok
```

Or compile to a standalone binary first:

```bash
./target/release/tok build examples/<file>.tok -o examples/<name>
./examples/<name>
```

If `tok` is on your PATH, you can simply use:

```bash
tok run examples/<file>.tok
```

### Using cargo (development)

During development you can use `cargo run` to build the compiler and run it in one step. The `--` separates cargo flags from arguments passed to `tok`:

```bash
cargo run -- run examples/<file>.tok
```

## Examples

### `language_tour.tok`
Core Tok syntax: variables, strings, arrays, maps, functions, lambdas, pipes, control flow (ternary, match, loops, FizzBuzz), tuples, error handling, destructuring, and higher-order functions.

```bash
tok run examples/language_tour.tok
```

### `math_and_algorithms.tok`
Math stdlib (`@"math"`) and classic algorithms: trigonometry, powers/logs, GCD/LCM, Newton's method for square roots, Collatz conjecture, Fibonacci, and Monte Carlo pi estimation.

```bash
tok run examples/math_and_algorithms.tok
```

### `word_counter.tok`
Text analysis: word splitting, frequency counting with `freq()`, map iteration, and basic statistics (longest word, average word length).

```bash
tok run examples/word_counter.tok
```

### `file_processor.tok`
CSV parsing and data transformation: splits CSV into structured maps, filters records, computes aggregate statistics, and outputs JSON via `@"json"`.

```bash
tok run examples/file_processor.tok
```

### `concurrency.tok`
Concurrency primitives: producer/consumer with buffered channels, parallel map (`pmap`), fan-out workers with goroutines, and multi-stage pipelines.

```bash
tok run examples/concurrency.tok
```

### `toon_data.tok`
TOON (Token-Oriented Object Notation) data format: parsing TOON documents with tabular arrays, filtering records, encoding values as TOON, and comparing JSON vs TOON output size. Demonstrates `@"toon"` and `@"json"` modules.

```bash
tok run examples/toon_data.tok
```

### `csv_data.tok`
CSV parsing and data analysis: parse CSV text into structured maps, filter by region, compute aggregate statistics, and encode filtered data back to CSV. Demonstrates `@"csv"` module.

```bash
tok run examples/csv_data.tok
```

### `template_demo.tok`
Template rendering with `{.key.}` syntax: variable substitution, loops over arrays, conditional and inverted sections, compile/apply for reuse, and nested data. Demonstrates `@"tmpl"` module.

```bash
tok run examples/template_demo.tok
```

### `llm_demo.tok`
LLM API calls: one-shot `ask`, multi-turn `chat`, system messages, and provider selection. Requires `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in environment. Demonstrates `@"llm"` module.

```bash
ANTHROPIC_API_KEY=sk-... tok run examples/llm_demo.tok
```

### `http_api.tok`
An HTTP server using `@"http"`. Starts on port 8080 with route handlers for GET and POST. Requires building separately since the server runs indefinitely.

```bash
# Terminal 1: build and start the server
tok build examples/http_api.tok -o examples/http_api
./examples/http_api

# Terminal 2: test the endpoints
curl http://localhost:8080/          # "Welcome to Tok API"
curl http://localhost:8080/health    # "ok"
curl -X POST http://localhost:8080/echo -d 'hello'  # echoes body
curl http://localhost:8080/anything  # 404 "not found"
```
