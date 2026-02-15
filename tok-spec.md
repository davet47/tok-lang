# Tok Language Specification v0.1

## 1. Design Philosophy

Tok is a general-purpose programming language designed for a single metric: **minimizing the number of tokens an LLM must generate** to express any given program. It compiles to native machine code via Cranelift.

### Non-goals
- Human readability
- Familiarity with existing languages
- Verbose error messages at the language syntax level

### Design principles
1. **Every character earns its place** — no boilerplate, no ceremony
2. **Common case is the default** — only unusual behavior requires annotation
3. **Symbols over words** — punctuation is typically 1 token; keywords are 1+ tokens
4. **Implicit over explicit** — returns, types, and scope are inferred
5. **Composition over nesting** — pipelines replace deeply nested calls
6. **Separators are optional where unambiguous** — newlines and semicolons both work; commas are never required

---

## 2. Lexical Structure

### 2.1 Source Encoding
UTF-8. No BOM.

### 2.2 Statement Separators
Statements are separated by **newlines** or **semicolons**. Both are equivalent. Multiple separators are collapsed (blank lines are fine).

```
x=1;y=2;z=x+y
```
is identical to:
```
x=1
y=2
z=x+y
```

### 2.3 Whitespace
Whitespace (spaces, tabs) is insignificant except:
- Inside strings
- As a separator between elements in arrays, maps, function params, and tuples

Indentation has no semantic meaning.

### 2.4 Comments
```
# This is a comment (to end of line)
```
No multi-line comment syntax. LLMs generating code rarely need comments.

### 2.5 Reserved Symbols & Keywords

**Keywords** (minimal set — 6 total):
| Keyword | Meaning |
|---------|---------|
| `f` | function declaration |
| `go` | spawn concurrent task |
| `sel` | select on channels |
| `T` | boolean true |
| `F` | boolean false |
| `N` | nil |

**Operator symbols** (all reserved):
```
=  +  -  *  /  %  **
== != > < >= <=
& | !
&& || ^^ << >>
?  ?= ?: ?^ ??
~ ^ .? ..
|> ?> />
# @
<- ->
```

**Structural symbols**:
```
( ) [ ] { } ; : " ` \ _ .
```

---

## 3. Values & Literals

### 3.1 Integers
Bare decimal numbers. Underscore separators allowed.
```
42
1_000_000
0xff          # hex
0b1010        # binary
0o77          # octal
```

### 3.2 Floats
Decimal point required. Scientific notation supported.
```
3.14
1.0e10
.5            # leading zero optional
```

### 3.3 Strings
Double-quoted. Interpolation with `{}`. Multi-line by default.
```
"hello"
"hello {name}"
"line 1
line 2"
```

**Raw strings** (no interpolation, no escapes): backtick-delimited.
```
`raw {not interpolated} \n literal`
```

**Escape sequences**: `\n` `\t` `\\` `\"` `\{` `\0` `\x41` `\u{1F600}`

### 3.4 Booleans
```
T    # true
F    # false
```

### 3.5 Nil
```
N
```

### 3.6 Arrays
Square brackets, space-separated. No commas.
```
[1 2 3]
["a" "b" "c"]
[1 "mixed" T N]
[[1 2] [3 4]]       # nested
[]                   # empty
```

### 3.7 Maps
Curly braces, `key:value` pairs, space-separated. No commas.
```
{a:1 b:2 c:3}
{name:"Alice" age:30}
{"key with spaces":42}
{}                   # empty
```

Keys are strings by default (bare identifiers are auto-stringified). Use `""` for keys with spaces or special chars.

### 3.8 Tuples
Parentheses, space-separated. Fixed-size, heterogeneous.
```
(1 "hello" T)
(200 N)              # common for result types: (value error)
```

### 3.9 Ranges
```
0..10               # 0 to 9 (exclusive end)
0..=10              # 0 to 10 (inclusive end)
10..0               # countdown: 10 to 1
```

---

## 4. Variables & Assignment

### 4.1 Declaration and Assignment
No keywords. Bare `name=value`.
```
x=5
name="Tok"
items=[1 2 3]
```

First assignment in a scope creates the variable. Subsequent assignments reassign.

### 4.2 Optional Type Annotations
Type annotations are optional — the compiler performs forward-flow type inference and defaults to `Any` when types can't be determined. Annotations can be used for documentation or to constrain types.

```
x:i=5                    # annotate as Int
y:f=3.14                 # annotate as Float
name:s="Alice"           # annotate as String
flag:b=T                 # annotate as Bool
data:a=get_data()        # annotate as Any (dynamic)
```

**Type annotation shorthands**:
| Annotation | Type |
|-----------|------|
| `:i` | Int |
| `:f` | Float |
| `:s` | String |
| `:b` | Bool |
| `:N` | Nil |
| `:a` | Any |
| `:[T]` | Array of T |
| `:{T}` | Map with values of type T |
| `:(T U V)` | Tuple of types |

Function parameters and return types can also be annotated:
```
f add(a:i b:i):i=a+b
```

### 4.3 Destructuring
```
a b=div(10 3)        # a=3, b=1 (tuple unpack)
{x y}=point          # map destructure
[h ..t]=list          # head/tail: h=first element, t=rest
_ e=might_fail()      # discard value, keep error
```

### 4.4 Compound Assignment
```
x+=1
x-=1
x*=2
x/=2
x%=2
x**=3
```

### 4.5 Scoping
Lexical scoping. Blocks `{}` create new scopes. Inner scopes can read and shadow outer variables.

---

## 5. Functions

### 5.1 Named Functions
```
f name(params)=expr          # single-expression body
f name(params){body}         # block body
```

Examples:
```
f add(a b)=a+b
f max(a b){a>b?a:b}
f greet(name)=p("hello {name}")
```

### 5.2 Implicit Return
The last expression in a block is the return value. No `return` keyword needed.
```
f abs(x){
  x<0?-x:x
}
```

### 5.3 Early Return: `^`
Use `^` to return early from a function. `^expr` returns `expr`.
```
f find(arr val){
  ~(i:0..#arr){
    arr[i]==val?^i
  }
  ^-1
}
```

Bare `^` returns `N` (nil).

### 5.4 Lambdas (Anonymous Functions)
```
\(params)=expr               # single-expression lambda
\(params){body}              # block lambda
\=expr                       # zero-parameter lambda
```

Examples:
```
sq=\(x)=x*x
add=\(a b)=a+b
thunk=\=42
```

### 5.5 Default Parameters
```
f greet(name msg="hello")="{msg} {name}"
```

### 5.6 Variadic Parameters
```
f sum(..nums)=nums/>\(a x)=a+x
```

`..` prefix collects remaining args into an array.

### 5.7 Function Calls
```
add(1 2)                     # basic call
p("hello")                   # stdlib call
obj.method()                 # method call
```

Arguments are space-separated. No commas.

### 5.8 Closures
All lambdas and named functions are closures — they capture their lexical environment.

---

## 6. Control Flow

### 6.1 Ternary Conditional
The primary conditional construct. Extremely token-efficient.
```
condition?then_expr:else_expr
```

**One-branch** (no else — evaluates to `N` if false):
```
condition?expr
```

**Block bodies**:
```
condition?{
  stmt1
  stmt2
}:{
  stmt3
}
```

Ternaries nest naturally:
```
x>0?"pos":x<0?"neg":"zero"
```

### 6.2 Multi-Branch Match: `?=`
Replaces if/elif/else chains and switch/case.
```
expr?={
  pattern1:result1
  pattern2:result2
  _:default_result
}
```

**Value matching**:
```
x?={
  0:"zero"
  1:"one"
  _:"other"
}
```

**Condition matching** (no expr before `?=`):
```
?={
  x>100:"big"
  x>10:"medium"
  x>0:"small"
  _:"non-positive"
}
```

**Pattern matching with destructure**:
```
result?={
  (v N):v
  (N e):{p("error: {e}");^}
}
```

**Block bodies in branches**:
```
cmd?={
  "start":{init();run()}
  "stop":{cleanup();exit()}
  _:p("unknown")
}
```

### 6.3 Loops: `~`

**While loop**:
```
~(condition){body}
```

**Range loop**:
```
~(i:0..10){p(i)}
```

**Foreach**:
```
~(item:array){p(item)}
```

**Foreach with index**:
```
~(i v:array){p("{i}: {v}")}
```

**Foreach key-value (maps)**:
```
~(k v:map){p("{k}={v}")}
```

**Infinite loop**:
```
~{body}
```

**Loop control**:
- `!` — break out of the innermost loop
- `>!` — continue to next iteration

```
~(i:0..100){
  i%2==0?>!          # skip even numbers
  i>50?!             # break at 50
  p(i)
}
```

**Loop as expression** (collects values):
```
squares=~(i:0..10)=i*i       # returns [0 1 4 9 16 25 36 49 64 81]
```

This replaces list comprehensions with no extra syntax.

---

## 7. Operators

### 7.1 Precedence Table (highest to lowest)

| Prec | Operators | Associativity | Description |
|------|-----------|---------------|-------------|
| 14 | `.` `.?` | left | member access |
| 13 | `#` | prefix | length |
| 12 | `!` `-` (unary) | prefix | not, negation |
| 11 | `**` | right | power |
| 10 | `*` `/` `%` | left | multiplicative |
| 9 | `+` `-` | left | additive |
| 8 | `..` `..=` | none | range |
| 7 | `<<` `>>` | left | bitwise shift |
| 6 | `&&` | left | bitwise and |
| 5 | `^^` | left | bitwise xor |
| 4 | `\|\|` | left | bitwise or |
| 3 | `==` `!=` `>` `<` `>=` `<=` | left | comparison |
| 2 | `&` | left | logical and |
| 1 | `\|` | left | logical or |
| 0 | `?` `?:` `??` `\|>` `?>` `/>` | right | conditionals, pipes |

### 7.2 Arithmetic
```
a+b   a-b   a*b   a/b   a%b   a**b
```

Division between integers produces integer (truncated). Use `float(a)/b` for float division, or ensure one operand is float: `a/1.0/b`.

### 7.3 Comparison
```
a==b   a!=b   a>b   a<b   a>=b   a<=b
```

### 7.4 Logical
```
a&b    # and (short-circuit)
a|b    # or (short-circuit)
!a     # not
```

### 7.5 Bitwise
```
a&&b   a||b   a^^b   a<<b   a>>b
```

Note: `&&` and `||` are bitwise; `&` and `|` are logical. This is the reverse of C/Java convention but saves tokens (logical ops are far more common, so they get the shorter symbol).

### 7.6 Pipeline: `|>`
```
x|>f              # f(x)
x|>f(y)           # f(x y)
x|>f|>g|>h        # h(g(f(x)))
```

Pipelines eliminate nesting and read left-to-right.

### 7.7 Filter: `?>`
```
arr?>\(x)=x>3     # keep elements where predicate is true
```

### 7.8 Reduce: `/>`
```
arr/>\(acc x)=acc+x    # fold/reduce
arr/>0 \(acc x)=acc+x  # with initial value
```

### 7.9 Length: `#`
Prefix operator. Works on arrays, strings, maps.
```
#[1 2 3]          # 3
#"hello"          # 5
#{a:1 b:2}        # 2
```

### 7.10 Spread: `..`
```
a=[1 2 3]
b=[0 ..a 4]       # [0 1 2 3 4]
f(..args)          # spread array as function arguments
{..defaults ..overrides}  # merge maps
```

### 7.11 Nil Handling
```
x??default         # nil coalesce: x if x!=N, else default
obj.?field         # optional chain: N if obj is N, else obj.field
obj.?method()      # optional call: N if obj is N, else obj.method()
```

---

## 8. Data Structures

### 8.1 Prototypes (Object System)
Tok uses prototype-based objects. No `class` keyword.

**Define a prototype**:
```
Point={
  x:0
  y:0
  dist:f()=(.x**2+.y**2)**0.5
  add:f(o)=Point{x:.x+o.x y:.y+o.y}
}
```

**Instantiate**:
```
p=Point{x:3 y:4}
p.dist()              # 5.0
```

**Self-reference**: `.field` inside a method refers to the current object's field. No `self`/`this` keyword needed.

### 8.2 Prototype Extension (Inheritance)
```
Point3D=Point{
  z:0
  dist:f()=(.x**2+.y**2+.z**2)**0.5
}
p3=Point3D{x:1 y:2 z:3}
```

### 8.3 Array Methods (via pipeline)
Arrays are operated on via pipeline operators and stdlib functions:
```
[3 1 4 1 5]|>sort                 # [1 1 3 4 5]
[3 1 4 1 5]|>sort|>rev            # [5 4 3 1 1]
[1 2 3]|>\(x)=x*2                # [2 4 6]  (map)
[1 2 3 4 5]?>\(x)=x>2            # [3 4 5]  (filter)
[1 2 3 4 5]/>\(a x)=a+x          # 15       (reduce)
```

### 8.4 Map Operations
```
m={a:1 b:2 c:3}
keys(m)                           # ["a" "b" "c"]
vals(m)                           # [1 2 3]
has(m "a")                        # T
del(m "a")                        # {b:2 c:3}
{..m d:4}                         # {a:1 b:2 c:3 d:4}
```

---

## 9. Error Handling

### 9.1 Result Tuples
Functions that can fail return `(value error)` tuples. Success: `(val N)`. Failure: `(N errMsg)`.

```
f div(a b){
  b==0?^(N "division by zero")
  (a/b N)
}
```

### 9.2 Unpacking Results
```
v e=div(10 3)
e?{p("error: {e}");^}
p(v)
```

### 9.3 Error Propagation: `?^`
Automatically unpacks a result tuple. If error is non-nil, returns the error from the current function.

```
f calc(){
  a=div(10 2)?^        # a=5, or propagate error
  b=div(a 0)?^         # propagates "division by zero"
  a+b
}
```

`?^` is equivalent to:
```
_tmp=div(10 2)
_tmp.1?^_tmp
_tmp.0
```

### 9.4 Error Recovery
```
v e=risky_op()
v=v??fallback_value
```

Or with `?=` matching:
```
risky_op()?={
  (v N):process(v)
  (N e):p("failed: {e}")
}
```

---

## 10. Modules & Imports

### 10.1 Import Syntax
```
@"modname"                    # import all exports into scope
m=@"modname"                  # import as namespace
{a b c}=@"modname"           # destructured import
```

### 10.2 File-Based Modules
Each `.tok` file is a module. Top-level bindings are exports by default. Prefix with `_` to make private.

```
# math.tok
f add(a b)=a+b               # exported
f sub(a b)=a-b               # exported
_cache={}                     # private
```

### 10.3 Standard Module Paths (Planned)
These standard library modules are planned but not yet implemented:
```
@"io"                         # I/O operations
@"fs"                         # filesystem
@"net"                        # networking
@"http"                       # HTTP client/server
@"json"                       # JSON encode/decode
@"math"                       # math functions
@"str"                        # string utilities
@"os"                         # OS interaction
@"time"                       # time/date
@"re"                         # regex
```

File-based modules (`@"path/to/file"`) work fully, including circular import detection.

---

## 11. Concurrency

### 11.1 Goroutines (OS Threads)
```
h=go{expensive_computation()}
result=<-h                    # block until done, get result
```

Goroutines spawn OS threads (not green threads). Each goroutine gets an independent copy of its scope.

### 11.2 Channels
```
c=chan()                      # unbuffered channel
c=chan(10)                    # buffered channel (size 10)

go{c<-compute()}             # send to channel
v=<-c                        # receive from channel
```

### 11.3 Select (Multi-Channel Wait)
```
sel{
  v=<-c1:{p("from c1: {v}")}
  v=<-c2:{p("from c2: {v}")}
  _:{p("default")}
}
```

### 11.4 Parallel Map
```
results=[1 2 3 4 5]|>pmap(\(x)=heavy(x))
```

`pmap` runs the function concurrently across all elements.

---

## 12. Standard Library

### 12.1 Core Builtins (Always Available)

| Function | Description | Example |
|----------|-------------|---------|
| `p(x)` | Print (no newline) | `p("hi")` |
| `pl(x)` | Print with newline | `pl("hi")` |
| `#x` | Length operator | `#[1 2 3]` → `3` |
| `int(x)` | Convert to integer | `int("42")` → `42` |
| `float(x)` | Convert to float | `float(42)` → `42.0` |
| `str(x)` | Convert to string | `str(42)` → `"42"` |
| `type(x)` | Get type as string | `type(42)` → `"int"` |
| `sort(a)` | Sort array | `sort([3 1 2])` → `[1 2 3]` |
| `rev(a)` | Reverse array | `rev([1 2 3])` → `[3 2 1]` |
| `keys(m)` | Map keys | `keys({a:1})` → `["a"]` |
| `vals(m)` | Map values | `vals({a:1})` → `[1]` |
| `has(m k)` | Map has key | `has({a:1} "a")` → `T` |
| `del(m k)` | Delete key from map | `del({a:1 b:2} "a")` → `{b:2}` |
| `push(a x)` | Append to array | `push([1 2] 3)` → `[1 2 3]` |
| `join(a s)` | Join array to string | `join(["a" "b"] ",")` → `"a,b"` |
| `split(s d)` | Split string | `split("a,b" ",")` → `["a" "b"]` |
| `trim(s)` | Trim whitespace | `trim(" hi ")` → `"hi"` |
| `slice(x i j)` | Slice array/string | `slice([1 2 3 4] 1 3)` → `[2 3]` |
| `flat(a)` | Flatten nested array | `flat([[1 2] [3]])` → `[1 2 3]` |
| `uniq(a)` | Unique elements | `uniq([1 1 2])` → `[1 2]` |
| `min(a)` | Minimum | `min([3 1 2])` → `1` |
| `max(a)` | Maximum | `max([3 1 2])` → `3` |
| `sum(a)` | Sum | `sum([1 2 3])` → `6` |
| `abs(x)` | Absolute value | `abs(-5)` → `5` |
| `floor(x)` | Floor | `floor(3.7)` → `3` |
| `ceil(x)` | Ceiling | `ceil(3.2)` → `4` |
| `rand()` | Random float 0..1 | `rand()` → `0.472...` |
| `exit(n)` | Exit with code | `exit(0)` |
| `clock()` | Current time (seconds) | `clock()` → `1234567.89` |
| `chan(n)` | Create channel | `chan()` or `chan(10)` |
| `pmap(a f)` | Parallel map | `pmap([1 2 3] \(x)=x*2)` |

### 12.2 Standard Library Modules (Not Yet Implemented)

The following module interfaces are planned but not yet available:

- `@"io"` — stdin/stdout operations (`input`, `readall`)
- `@"fs"` — filesystem (`fread`, `fwrite`, `fexists`, `fls`, etc.)
- `@"http"` — HTTP client/server (`hget`, `hpost`, `serve`, etc.)
- `@"json"` — JSON encode/decode (`jparse`, `jstr`)
- `@"re"` — regex (`rmatch`, `rfind`, `rall`, `rsub`)
- `@"time"` — time/date (`now`, `sleep`, `fmt`)
- `@"math"` — extended math functions
- `@"str"` — extended string utilities
- `@"os"` — OS interaction

Additional builtins planned but not yet implemented: `is(x t)`, `args()`, `env(k)`, `pop(a)`, `freq(a)`, `top(m n)`, `zip(a b)`.

---

## 13. Memory Model

### 13.1 Automatic Memory Management
Tok uses **reference counting** with `AtomicU32` for all heap-allocated types. The programmer never explicitly allocates or frees memory. Allocated via `Box::into_raw`, freed via `Box::from_raw` when the refcount reaches 0.

**Limitations**: No cycle detection — reference cycles will leak memory.

### 13.2 Value Semantics
- Integers, floats, booleans, nil: value types (stack-allocated, unboxed in Cranelift IR)
- Strings: immutable, reference counted
- Arrays, maps, tuples, closures, channels, handles: heap-allocated, reference counted

---

## 14. Compilation Model

### 14.1 Pipeline
```
Tok source → Lexer → Parser → AST → Type Inference → HIR → Cranelift IR → .o → Native Binary
```

The compiler is structured as 7 Rust crates: `tok-lexer`, `tok-parser`, `tok-types`, `tok-hir`, `tok-codegen`, `tok-runtime`, and `tok-driver` (CLI). The final linking step uses `cc` to link the `.o` file with the static C runtime library (`libtok_runtime.a`).

### 14.2 Type System Strategy
- **Optionally typed**: type annotations are available (`:i`, `:f`, `:s`, `:b`, `:a`) but never required — fewer tokens by default
- **Forward-flow type inference** (not Hindley-Milner): types propagate forward through assignments and expressions
- **Lenient**: emits warnings instead of errors, defaults to `Any` when types can't be determined
- **Statically-typed values are unboxed**: Int=`i64`, Float=`f64`, Bool=`i8`, heap types=`i64` pointer
- **`Any` type**: uses `TokValue`, a 16-byte tagged union (`tag:u8` + `data:union{i64, f64, i8, *mut T}`) stored on the stack
- **Mixed-type branches**: when if/else branches have different types, the merge block upgrades to `Any` semantics

Key unification rules:
| Unification | Result |
|---|---|
| `unify(Any, X)` | `Any` |
| `unify(Int, Float)` | `Float` |
| `unify(Nil, T)` | `Optional(T)` |
| `unify(X, X)` | `X` |
| incompatible | `Any` |

### 14.3 Runtime Components
- Reference-counted heap allocations (`AtomicU32` refcounts)
- Dynamic dispatch for `Any`-typed values (tag-based)
- OS threads for goroutines (1:1 threading model)
- Channel implementation (Mutex + Condvar, both buffered and unbuffered)
- Static C-ABI runtime library linked into every binary

---

## 15. Complete Syntax Quick Reference

```
# Variables
x=5; s="hi"; a=[1 2 3]; m={k:v}
x:i=5; y:f=3.14              # optional type annotations

# Functions
f name(p1 p2)=expr
f name(p1 p2){body}
f add(a:i b:i):i=a+b         # with type annotations
\(p)=expr                    # lambda

# Control
cond?then:else               # ternary
expr?={pat:res _:def}        # match
~(cond){body}                # while
~(v:arr){body}               # foreach
~(i:0..n){body}              # range for

# Operators
|> pipe   ?> filter   /> reduce
#  length  .. spread  ?? nil-coalesce
^  return  !  break   >! continue

# Concurrency
h=go{expr}; v=<-h            # spawn & await
c=chan(); c<-v; v=<-c         # channels
sel{v=<-c1:{...} _:{...}}    # select

# Modules
@"mod"; m=@"mod"; {a b}=@"mod"

# Error handling
v e=might_fail()              # unpack result
v=might_fail()?^              # propagate error

# Keywords (6 total)
f  go  sel  T  F  N
```
