# Tok

A programming language designed for one metric: **minimizing the number of tokens an LLM must generate** to express any program. Compiles to native machine code via Cranelift.

```tok
f fib(n)=n<2?n:fib(n-1)+fib(n-2)
pl(fib(10))
```

```tok
~(i:1..101){pl(?={i%15==0:"FizzBuzz";i%3==0:"Fizz";i%5==0:"Buzz";_:i})}
```

```tok
f qsort(a){
  (#a)<=1?^a
  p=a[(#a)/2]
  lo=a?>\(x)=x<p
  eq=a?>\(x)=x==p
  hi=a?>\(x)=x>p
  [..qsort(lo) ..eq ..qsort(hi)]
}
```

## Design Principles

- **Every character earns its place** -- no boilerplate, no ceremony
- **Symbols over words** -- 5 keywords total: `f` `T` `F` `N` `go`
- **Implicit over explicit** -- returns, types, and scope are inferred
- **Composition over nesting** -- pipelines replace deeply nested calls
- **No commas** -- spaces separate elements everywhere

## Quick Start

```bash
# Build
cargo build

# Compile and run a Tok program
cargo run -- run examples.tok

# Just compile (produces a native binary)
cargo run -- build examples.tok -o examples
./examples
```

Other commands: `lex` (print tokens), `parse` (print AST), `check` (type check).

## Language Overview

### Basics

```tok
x=42                    # variables (no let/var/const)
name="Alice"            # strings
nums=[1 2 3 4 5]       # arrays (no commas)
m={name:"Alice" age:30} # maps
t=(1 "hello" T)         # tuples
```

### Functions

```tok
f add(a b)=a+b                    # one-line function
f greet(name)="Hello, {name}!"   # string interpolation

f factorial(n){                    # block body
  n<=1?^1                         # conditional return: ?^
  n*factorial(n-1)                # implicit return (last expr)
}
```

### Control Flow

```tok
# Ternary
x=age>=18?"adult":"minor"

# Match
grade=?={score>=90:"A";score>=80:"B";score>=70:"C";_:"F"}

# Loops
~(i:0..10){pl(i)}        # range loop
~(x:arr){pl(x)}          # foreach
~(cond){body}             # while
```

### Operators

```tok
nums|>sort|>rev           # pipeline: x|>f  =  f(x)
evens=nums?>\(x)=x%2==0  # filter
total=nums/>0 \(a x)=a+x # reduce with init value
#arr                      # length
..arr                     # spread
x??default                # nil coalesce
obj.?field                # optional chaining
```

### Lambdas & Closures

```tok
double=\(x)=x*2
add=\(a b)=a+b
[1 2 3]?>\(x)=x>1        # filter with lambda
```

### Error Handling

```tok
# Result tuples: (value error)
f safe_div(a b)=b==0?(N "division by zero"):(a/b N)

result=safe_div(10 0)
val err=result              # tuple destructuring
pl(val??0)                  # nil coalesce for default
```

### Concurrency

```tok
ch=chan()                   # create channel
h=go{expensive_work()}     # spawn goroutine
ch<-42                      # send to channel
val=<-ch                    # receive from channel

results=pmap(items \(x)=process(x))  # parallel map
```

### Modules

```tok
@"utils"                    # import and merge into scope
m=@"helpers"                # import as map
pl(m.some_func(42))
```

## Architecture

The compiler is structured as a pipeline of 7 Rust crates:

```
Source Text
    |
    v
 [tok-lexer]     Tokenization
    |
    v
 [tok-parser]    Recursive-descent parsing -> AST
    |
    v
 [tok-types]     Forward-flow type inference
    |
    v
 [tok-hir]       Desugaring -> typed IR
    |
    v
 [tok-codegen]   Cranelift IR generation -> .o file
    |
    v
 [tok-runtime]   Static C runtime library (libtok_runtime.a)
    |
    v
 [linker]        cc links .o + runtime -> native binary
```

All orchestrated by **tok-driver**, the CLI entry point.

### Crate Dependency Graph

```
tok-driver
  +-- tok-lexer
  +-- tok-parser        --> tok-lexer
  +-- tok-types         --> tok-parser
  +-- tok-hir           --> tok-parser, tok-types
  +-- tok-codegen       --> tok-hir, tok-types, tok-runtime, cranelift-*
```

`tok-runtime` has no internal dependencies and compiles as both a Rust library and a C static library (`libtok_runtime.a`).

### Crate Details

#### tok-lexer

Converts source text into a flat token stream. Handles string interpolation by splitting `"hello {expr}"` into `StringStart` / expr tokens / `StringEnd` sequences. Context-sensitive `#`: comment at line start, length operator otherwise.

#### tok-parser

Recursive-descent parser with 14 levels of operator precedence. Produces an AST with ~15 expression kinds and ~12 statement kinds. Notable disambiguation: `{` could start a map literal or a block -- resolved by lookahead for `IDENT:` or `STRING:` pattern.

#### tok-types

Forward-flow type inference (not Hindley-Milner). Lenient: emits warnings instead of errors, defaults to `Any` when types can't be determined. Key unification rules:

| Unification | Result |
|---|---|
| `unify(Any, X)` | `Any` |
| `unify(Int, Float)` | `Float` |
| `unify(Nil, T)` | `Optional(T)` |
| `unify(X, X)` | `X` |
| incompatible | `Any` |

#### tok-hir

Lowers the AST into a simpler, type-annotated intermediate representation. Every `HirExpr` node carries its inferred `Type`. Desugars: compound assignments (`+=`), string interpolation (to concat calls), pipelines (to nested calls), destructuring, nil coalesce / error propagation (to if-else chains), match (to if-else chains).

#### tok-codegen

Translates HIR to Cranelift IR and emits a `.o` object file. Statically-typed values are unboxed (Int=i64, Float=f64, Bool=i8, pointers for heap types). The `Any` type uses a 16-byte tagged union (`TokValue`) with tag + data stored on the stack.

Key design decisions:
- **Two-pass compilation**: functions first, then deferred lambdas
- **Closure captures**: free variable analysis, heap-allocated environment buffers
- **Mixed-type branches**: when if/else branches have different types, the merge block uses `Any` semantics and wraps/unwraps as needed

#### tok-runtime

The C-ABI runtime library linked into every compiled binary. All heap types are reference-counted with `AtomicU32`. Provides:

| Category | Functions |
|---|---|
| **I/O** | `p`, `pl` (print/println with type dispatch) |
| **Arrays** | alloc, push, get, set, len, sort, rev, flat, uniq, slice, filter, reduce, concat, min, max, sum |
| **Strings** | alloc, concat, len, index, slice, split, trim, join, eq, cmp |
| **Maps** | alloc, get, set, has, del, keys, vals, len (ordered `Vec`, not HashMap) |
| **Tuples** | alloc, get, set, len |
| **Closures** | alloc with fn_ptr + env_ptr, env heap allocation |
| **Channels** | alloc (buffered/unbuffered), send, recv, try_send, try_recv |
| **Goroutines** | go (spawn thread), handle join |
| **Math** | abs, floor, ceil, rand + dynamic-dispatch variants |
| **Conversions** | int, float, str, type_of, to_string |
| **Dynamic ops** | add, sub, mul, div, mod, negate, eq, lt, truthiness (for `Any` type) |

### Value Representation

Static types compile to unboxed Cranelift values:

| Tok Type | Cranelift | Size |
|---|---|---|
| Int | `i64` | 8 bytes |
| Float | `f64` | 8 bytes |
| Bool | `i8` | 1 byte |
| Nil | -- | 0 (no representation) |
| String, Array, Map, Tuple, Closure, Channel, Handle | `i64` (pointer) | 8 bytes |

The `Any` type uses `TokValue`, a 16-byte tagged union (`#[repr(C)]`):

```
+--------+--------+
| tag(8) | data(8)|
+--------+--------+
  u8+pad   union{ i64, f64, i8, *mut T }
```

Tags: 0=Nil, 1=Int, 2=Float, 3=Bool, 4=String, 5=Array, 6=Map, 7=Tuple, 8=Func, 9=Channel, 10=Handle.

## What's Implemented

All 9 phases of the language spec are complete:

| Phase | Features | Status |
|---|---|---|
| 1 | Literals, variables, operators, arrays, maps | Done |
| 2 | Ternary, loops (range/while/foreach), match | Done |
| 3 | Functions, lambdas, closures, implicit returns | Done |
| 4 | Strings, interpolation, escape sequences | Done |
| 5 | Pipes (`\|>`), filter (`?>`), reduce (`/>`) | Done |
| 6 | Maps, member access, nested maps, builtins (sort, rev, keys, vals, etc.) | Done |
| 7 | Modules (`@"path"`), circular import detection | Done |
| 8 | Tuples, destructuring, error propagation (`?^`), nil coalesce (`??`), optional chain (`.?`) | Done |
| 9 | Goroutines (`go`), channels, select (`sel`), parallel map (`pmap`) | Done |

**Compiler backend**: Cranelift AOT compilation to native binaries. All 7 end-to-end test suites pass.

## What's Not Yet Implemented

- **`slice` builtin** in the compiler (works in interpreter, not yet in codegen)
- **Destructured imports**: `{a b c}=@"path"` syntax
- **Standard library modules**: `@"io"`, `@"math"`, `@"http"`, `@"json"`, etc.
- **Interpreter mode**: The project currently only has the native compiler path; no `eval`/REPL
- **Concurrency in codegen**: `go`, channels, `sel`, and `pmap` work in the interpreter but are not yet compiled to native code
- **Garbage collection**: Currently uses reference counting only (no cycle detection)

## Tests

```bash
# Unit tests (246 tests across all crates)
cargo test --workspace

# End-to-end compiler tests (compile + run each test file)
cargo run -- run tests/basics_test.tok
cargo run -- run tests/control_flow_test.tok
cargo run -- run tests/functions_test.tok
cargo run -- run tests/arrays_lambdas_test.tok
cargo run -- run tests/strings_pipes_test.tok
cargo run -- run tests/maps_test.tok
cargo run -- run tests/errors_tuples_test.tok
```

## Project Structure

```
tok-lang/
  Cargo.toml              # Workspace root
  tok-spec.md             # Language specification
  tok-grammar.ebnf        # Formal grammar
  tok-examples.md         # Example programs with comparisons
  crates/
    tok-lexer/            # Tokenizer
    tok-parser/           # Parser -> AST
    tok-types/            # Type inference
    tok-hir/              # HIR lowering
    tok-codegen/          # Cranelift code generation
    tok-runtime/          # C-ABI runtime library
    tok-driver/           # CLI binary
  tests/
    *_test.tok            # End-to-end test files
    bench/                # Benchmark programs (Tok vs Go vs Rust)
```
