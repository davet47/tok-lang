#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/../.."
TOK="$ROOT_DIR/target/release/tok"
BUILD_DIR="$SCRIPT_DIR/build"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

mkdir -p "$BUILD_DIR"

echo -e "${YELLOW}=== Building ===${NC}"

# Build Tok (release)
echo -n "  Tok (cargo build --release)... "
(cd "$ROOT_DIR" && cargo build --release 2>/dev/null)
echo -e "${GREEN}done${NC}"

# Compile Rust benchmarks
for f in "$SCRIPT_DIR"/*.rs; do
    name=$(basename "$f" .rs)
    echo -n "  Rust ($name)... "
    rustc -O -o "$BUILD_DIR/${name}_rust" "$f" 2>/dev/null
    echo -e "${GREEN}done${NC}"
done

# Compile Go benchmarks
for f in "$SCRIPT_DIR"/*.go; do
    name=$(basename "$f" .go)
    echo -n "  Go ($name)... "
    go build -o "$BUILD_DIR/${name}_go" "$f" 2>/dev/null
    echo -e "${GREEN}done${NC}"
done

echo ""
echo -e "${YELLOW}=== Running Benchmarks ===${NC}"
echo ""

for num in 01 02 03 04 05 06 07 08; do
    # Find the tok file for this benchmark number
    tok_file=$(ls "$SCRIPT_DIR"/${num}_*.tok 2>/dev/null | head -1 || true)
    if [ -z "$tok_file" ]; then continue; fi
    base=$(basename "$tok_file" .tok)

    echo -e "${CYAN}--- $base ---${NC}"
    echo ""

    # Run Tok
    echo -e "  ${BLUE}Tok:${NC}"
    "$TOK" "$tok_file" 2>&1 | sed 's/^/    /'
    echo ""

    # Run Rust
    rust_bin="$BUILD_DIR/${base}_rust"
    if [ -f "$rust_bin" ]; then
        echo -e "  ${BLUE}Rust:${NC}"
        "$rust_bin" 2>&1 | sed 's/^/    /'
        echo ""
    fi

    # Run Go
    go_bin="$BUILD_DIR/${base}_go"
    if [ -f "$go_bin" ]; then
        echo -e "  ${BLUE}Go:${NC}"
        "$go_bin" 2>&1 | sed 's/^/    /'
        echo ""
    fi
done

echo -e "${YELLOW}=== Done ===${NC}"
