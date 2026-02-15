// Benchmark 05: Maps — build, lookup, membership
// Measures map insertion, O(1) lookup, membership check
package main

import (
	"fmt"
	"time"
)

func main() {
	// Part A: Build map
	nBuild := 500_000
	start := time.Now()
	m := make(map[string]int64)
	for i := int64(0); i < int64(nBuild); i++ {
		m[fmt.Sprintf("k%d", i)] = i
	}
	elapsed := time.Since(start).Milliseconds()
	fmt.Printf("[bench] map_build: %d entries in %d ms\n", nBuild, elapsed)

	// Part B: Lookup every key
	start = time.Now()
	total := int64(0)
	for i := int64(0); i < int64(nBuild); i++ {
		total += m[fmt.Sprintf("k%d", i)]
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] map_lookup: %d lookups in %d ms (sum=%d)\n", nBuild, elapsed, total)

	// Part C: Membership check
	nHas := 1_000_000
	start = time.Now()
	found := int64(0)
	for i := int64(0); i < int64(nHas); i++ {
		k := fmt.Sprintf("k%d", i)
		if _, ok := m[k]; ok {
			found++
		}
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] map_has: %d checks in %d ms (found=%d)\n", nHas, elapsed, found)
}
