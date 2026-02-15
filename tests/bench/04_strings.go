// Benchmark 04: Strings — concatenation, format, split/join
// Measures string allocation, formatting overhead, string ops
package main

import (
	"fmt"
	"strings"
	"time"
)

func main() {
	// Part A: Concatenation
	nConcat := 20_000_000
	start := time.Now()
	var builder strings.Builder
	for i := 0; i < nConcat; i++ {
		builder.WriteByte('a')
	}
	s := builder.String()
	elapsed := time.Since(start).Milliseconds()
	fmt.Printf("[bench] string_concat: %d iterations in %d ms (len=%d)\n", nConcat, elapsed, len(s))

	// Part B: Interpolation (Sprintf)
	nInterp := 20_000_000
	start = time.Now()
	last := ""
	for i := 0; i < nInterp; i++ {
		last = fmt.Sprintf("%d", i)
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] string_interp: %d iterations in %d ms (last=%s)\n", nInterp, elapsed, last)

	// Part C: Split + Join
	base := "the quick brown fox jumps over the lazy dog"
	nSplit := 1_000_000
	start = time.Now()
	result := ""
	for i := 0; i < nSplit; i++ {
		parts := strings.Split(base, " ")
		result = strings.Join(parts, "-")
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] string_split_join: %d iterations in %d ms (result=%s)\n", nSplit, elapsed, result)
}
