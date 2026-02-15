// Benchmark 03: Arrays — build, sort, filter, reduce
// Measures array allocation, sorting, functional operations
package main

import (
	"fmt"
	"sort"
	"time"
)

func main() {
	// Part A: Build array via append
	nBuild := 2_000_000
	start := time.Now()
	a := make([]int64, 0)
	for i := int64(0); i < int64(nBuild); i++ {
		a = append(a, i)
	}
	elapsed := time.Since(start).Milliseconds()
	fmt.Printf("[bench] array_build: %d elements in %d ms\n", nBuild, elapsed)

	// Part B: Sort (reversed array)
	b := make([]int64, len(a))
	for i, v := range a {
		b[len(a)-1-i] = v
	}
	n := len(b)
	start = time.Now()
	sort.Slice(b, func(i, j int) bool { return b[i] < b[j] })
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] array_sort: %d elements in %d ms (first=%d last=%d)\n", n, elapsed, b[0], b[n-1])

	// Part C: Filter + Reduce
	start = time.Now()
	total := int64(0)
	for _, x := range a {
		if x%2 == 0 {
			total += x
		}
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] array_filter_reduce: %d elements in %d ms (result=%d)\n", len(a), elapsed, total)
}
