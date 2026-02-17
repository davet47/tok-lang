# Tok Token Efficiency Benchmark

Measures LLM token counts for equivalent programs in Tok, Python, Go, and JavaScript using [tiktoken](https://github.com/openai/tiktoken) (GPT-4's tokenizer).

## Setup

```bash
pip install tiktoken
```

## Usage

```bash
# Default table output
python tools/token_bench/token_count.py

# Detailed per-benchmark breakdown
python tools/token_bench/token_count.py --detail

# ASCII bar charts
python tools/token_bench/token_count.py --chart

# JSON output
python tools/token_bench/token_count.py --json

# Use GPT-4o tokenizer instead
python tools/token_bench/token_count.py --encoding o200k_base

# Count with comments included
python tools/token_bench/token_count.py --no-strip

# Count tokens for a single file
python tools/token_bench/token_count.py --single path/to/file.tok
```

## Benchmarks

| # | Benchmark | What it tests |
|---|-----------|--------------|
| 1 | Fibonacci | Function definition, recursion, ternary |
| 2 | Quicksort | Filter, spread, length operator, early return |
| 3 | Data Pipeline | Maps, closures, filter/reduce, match, string interpolation |
| 4 | Concurrency | Goroutines, channels, pmap vs threading boilerplate |
| 5 | HTTP API | Routing DSL compactness |
| 6 | Word Frequency | Pipelines, builtins, reduce |

## Methodology

- Comments and blank lines are stripped before counting (use `--no-strip` to include them)
- All programs implement the same algorithm in each language's idiomatic style
- Tokenizer: `cl100k_base` (GPT-4) by default
