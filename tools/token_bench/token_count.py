#!/usr/bin/env python3
"""
Tok Token Efficiency Benchmark

Compares LLM token counts for equivalent programs written in Tok vs
Python, Go, JavaScript, and C#. Proves Tok's core value proposition:
fewer tokens = cheaper and faster LLM code generation.

Usage:
    python tools/token_bench/token_count.py
    python tools/token_bench/token_count.py --detail
    python tools/token_bench/token_count.py --chart
    python tools/token_bench/token_count.py --json
    python tools/token_bench/token_count.py --no-strip
    python tools/token_bench/token_count.py --encoding o200k_base
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not installed. Run: pip install tiktoken", file=sys.stderr)
    sys.exit(1)


LANG_EXTENSIONS = {
    ".tok": "Tok",
    ".py": "Python",
    ".go": "Go",
    ".js": "JavaScript",
    ".cs": "C#",
}

LANG_ORDER = ["Tok", "Python", "Go", "JavaScript", "C#"]


def strip_comments(source: str, ext: str) -> str:
    """Remove comment-only lines and blank lines to compare only program logic."""
    lines = source.split("\n")
    stripped = []
    for line in lines:
        trimmed = line.strip()
        if not trimmed:
            continue
        if ext in (".tok", ".py") and trimmed.startswith("#"):
            continue
        if ext in (".tok", ".go", ".js", ".cs") and trimmed.startswith("//"):
            continue
        stripped.append(line)
    return "\n".join(stripped)


def load_programs(programs_dir: Path, do_strip: bool) -> dict:
    """Load all benchmark programs grouped by benchmark name."""
    benchmarks = {}

    for bench_dir in sorted(programs_dir.iterdir()):
        if not bench_dir.is_dir():
            continue

        name = bench_dir.name
        # Strip numeric prefix for display: "01_fibonacci" -> "Fibonacci"
        display_name = name.split("_", 1)[1].replace("_", " ").title() if "_" in name else name

        benchmarks[display_name] = {}
        for f in sorted(bench_dir.iterdir()):
            if f.suffix in LANG_EXTENSIONS:
                source = f.read_text()
                if do_strip:
                    source = strip_comments(source, f.suffix)
                lang = LANG_EXTENSIONS[f.suffix]
                benchmarks[display_name][lang] = {
                    "source": source,
                    "file": str(f.relative_to(programs_dir)),
                    "chars": len(source),
                    "lines": len([l for l in source.split("\n") if l.strip()]),
                }

    return benchmarks


def count_tokens(benchmarks: dict, encoding) -> dict:
    """Add token counts to all loaded programs."""
    for bench_name, langs in benchmarks.items():
        for lang, info in langs.items():
            info["tokens"] = len(encoding.encode(info["source"]))
    return benchmarks


def savings(tok_count: int, other_count: int) -> float:
    """Calculate percentage savings (negative = Tok uses fewer)."""
    if other_count == 0:
        return 0.0
    return ((tok_count - other_count) / other_count) * 100


def format_table(benchmarks: dict, encoding_name: str) -> str:
    """Generate a Markdown comparison table."""
    lines = []
    lines.append(f"## LLM Token Efficiency: Tok vs Other Languages")
    lines.append(f"")
    lines.append(f"Tokenizer: `{encoding_name}` (GPT-4)")
    lines.append(f"")

    # Header
    header = "| Benchmark | Tok | Python | Go | JS | C# | vs Python | vs Go | vs JS | vs C# |"
    sep = "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
    lines.append(header)
    lines.append(sep)

    totals = {lang: 0 for lang in LANG_ORDER}

    for bench_name, langs in benchmarks.items():
        tok_tokens = langs.get("Tok", {}).get("tokens", 0)
        py_tokens = langs.get("Python", {}).get("tokens", 0)
        go_tokens = langs.get("Go", {}).get("tokens", 0)
        js_tokens = langs.get("JavaScript", {}).get("tokens", 0)
        cs_tokens = langs.get("C#", {}).get("tokens", 0)

        totals["Tok"] += tok_tokens
        totals["Python"] += py_tokens
        totals["Go"] += go_tokens
        totals["JavaScript"] += js_tokens
        totals["C#"] += cs_tokens

        vs_py = f"{savings(tok_tokens, py_tokens):+.1f}%" if py_tokens else "—"
        vs_go = f"{savings(tok_tokens, go_tokens):+.1f}%" if go_tokens else "—"
        vs_js = f"{savings(tok_tokens, js_tokens):+.1f}%" if js_tokens else "—"
        vs_cs = f"{savings(tok_tokens, cs_tokens):+.1f}%" if cs_tokens else "—"

        lines.append(
            f"| {bench_name} | {tok_tokens} | {py_tokens} | {go_tokens} | {js_tokens} | {cs_tokens} | {vs_py} | {vs_go} | {vs_js} | {vs_cs} |"
        )

    # Totals row
    vs_py = f"**{savings(totals['Tok'], totals['Python']):+.1f}%**"
    vs_go = f"**{savings(totals['Tok'], totals['Go']):+.1f}%**"
    vs_js = f"**{savings(totals['Tok'], totals['JavaScript']):+.1f}%**"
    vs_cs = f"**{savings(totals['Tok'], totals['C#']):+.1f}%**"
    lines.append(
        f"| **Total** | **{totals['Tok']}** | **{totals['Python']}** | **{totals['Go']}** | **{totals['JavaScript']}** | **{totals['C#']}** | {vs_py} | {vs_go} | {vs_js} | {vs_cs} |"
    )

    lines.append("")

    # Summary line
    other_avg = (totals["Python"] + totals["Go"] + totals["JavaScript"] + totals["C#"]) / 4
    avg_savings = savings(totals["Tok"], other_avg)
    lines.append(
        f"**Tok uses {abs(avg_savings):.0f}% fewer tokens on average** across all benchmarks."
    )

    return "\n".join(lines)


def format_detail(benchmarks: dict) -> str:
    """Generate detailed per-benchmark breakdown."""
    lines = []
    for bench_name, langs in benchmarks.items():
        lines.append(f"\n### {bench_name}")
        tok_tokens = langs.get("Tok", {}).get("tokens", 0)
        best_other = float("inf")

        for lang in LANG_ORDER:
            info = langs.get(lang)
            if not info:
                continue
            marker = " <-- " if lang == "Tok" else ""
            lines.append(
                f"  {lang:12s} {info['tokens']:4d} tokens  {info['chars']:5d} chars  {info['lines']:3d} lines{marker}"
            )
            if lang != "Tok" and info["tokens"] < best_other:
                best_other = info["tokens"]

        if tok_tokens and best_other < float("inf"):
            pct = savings(tok_tokens, best_other)
            lines.append(f"  Savings vs next-best: {pct:+.1f}%")

    return "\n".join(lines)


def format_chart(benchmarks: dict) -> str:
    """Generate ASCII bar charts."""
    lines = []
    max_tokens = 0
    for langs in benchmarks.values():
        for info in langs.values():
            max_tokens = max(max_tokens, info.get("tokens", 0))

    bar_width = 50

    for bench_name, langs in benchmarks.items():
        lines.append(f"\n  {bench_name}")
        for lang in LANG_ORDER:
            info = langs.get(lang)
            if not info:
                continue
            tokens = info["tokens"]
            bar_len = int((tokens / max_tokens) * bar_width) if max_tokens else 0
            bar = "\u2588" * bar_len
            pad = " " * (bar_width - bar_len)
            lines.append(f"    {lang:12s} {bar}{pad} {tokens:4d}")

    return "\n".join(lines)


def format_json(benchmarks: dict, encoding_name: str) -> str:
    """Generate JSON output."""
    output = {"encoding": encoding_name, "benchmarks": {}}
    for bench_name, langs in benchmarks.items():
        output["benchmarks"][bench_name] = {}
        for lang, info in langs.items():
            output["benchmarks"][bench_name][lang] = {
                "tokens": info["tokens"],
                "chars": info["chars"],
                "lines": info["lines"],
                "file": info["file"],
            }
    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Tok LLM Token Efficiency Benchmark"
    )
    parser.add_argument(
        "--programs",
        default=None,
        help="Path to programs directory",
    )
    parser.add_argument("--detail", action="store_true", help="Show per-file details")
    parser.add_argument("--chart", action="store_true", help="Show ASCII bar charts")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--no-strip",
        action="store_true",
        help="Don't strip comments before counting",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="tiktoken encoding (default: cl100k_base)",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="Count tokens for a single file",
    )

    args = parser.parse_args()

    encoding = tiktoken.get_encoding(args.encoding)

    # Single file mode
    if args.single:
        source = Path(args.single).read_text()
        ext = Path(args.single).suffix
        if not args.no_strip:
            source = strip_comments(source, ext)
        tokens = len(encoding.encode(source))
        chars = len(source)
        line_count = len([l for l in source.split("\n") if l.strip()])
        print(f"{tokens} tokens, {chars} chars, {line_count} lines ({args.encoding})")
        return

    # Find programs directory
    if args.programs:
        programs_dir = Path(args.programs)
    else:
        # Try relative to script location
        script_dir = Path(__file__).parent
        programs_dir = script_dir / "programs"
        if not programs_dir.exists():
            # Try relative to cwd
            programs_dir = Path("tools/token_bench/programs")

    if not programs_dir.exists():
        print(f"Error: programs directory not found: {programs_dir}", file=sys.stderr)
        sys.exit(1)

    benchmarks = load_programs(programs_dir, do_strip=not args.no_strip)
    benchmarks = count_tokens(benchmarks, encoding)

    if args.json:
        print(format_json(benchmarks, args.encoding))
    else:
        print(format_table(benchmarks, args.encoding))
        if args.detail:
            print(format_detail(benchmarks))
        if args.chart:
            print(format_chart(benchmarks))


if __name__ == "__main__":
    main()
