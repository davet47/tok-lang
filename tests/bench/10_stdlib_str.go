// Benchmark 10: Stdlib Str — upper/lower, contains, replace, split
// Measures string manipulation overhead

package main

import (
	"fmt"
	"strings"
	"time"
)

func main() {
	// Part A: upper + lower case conversion loop
	nCase := 2_000_000
	start := time.Now()
	last := ""
	for i := 0; i < nCase; i++ {
		last = strings.ToUpper("hello world")
		last = strings.ToLower(last)
	}
	elapsed := time.Since(start).Milliseconds()
	fmt.Printf("[bench] str_case: %d upper+lower pairs in %d ms (last=%s)\n", nCase, elapsed, last)

	// Part B: contains + starts_with + ends_with checks
	base := "the quick brown fox jumps over the lazy dog"
	nSearch := 2_000_000
	start = time.Now()
	count := 0
	for i := 0; i < nSearch; i++ {
		if strings.Contains(base, "fox") {
			count++
		}
		if strings.HasPrefix(base, "the") {
			count++
		}
		if strings.HasSuffix(base, "dog") {
			count++
		}
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] str_search: %d x3 checks in %d ms (count=%d)\n", nSearch, elapsed, count)

	// Part C: replace
	nReplace := 1_000_000
	start = time.Now()
	last2 := ""
	for i := 0; i < nReplace; i++ {
		last2 = strings.ReplaceAll("hello world hello", "hello", "hi")
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] str_replace: %d calls in %d ms (last=%s)\n", nReplace, elapsed, last2)

	// Part D: split + rejoin
	nSplit := 1_000_000
	start = time.Now()
	last3 := ""
	for i := 0; i < nSplit; i++ {
		parts := strings.Split("a,b,c,d,e", ",")
		last3 = strings.Join(parts, "-")
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] str_split: %d split+join in %d ms (last=%s)\n", nSplit, elapsed, last3)
}
