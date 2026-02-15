// Benchmark 02: Loops — iteration overhead
// Measures counting loop, arithmetic body, while loop
package main

import (
	"fmt"
	"time"
)

func main() {
	// Part A: Counting loop
	nCount := int64(5_000_000_000)
	start := time.Now()
	x := int64(0)
	for i := int64(0); i < nCount; i++ {
		x++
	}
	elapsed := time.Since(start).Milliseconds()
	fmt.Printf("[bench] loops_count: %d iterations in %d ms (result=%d)\n", nCount, elapsed, x)

	// Part B: Arithmetic body
	nArith := int64(1_000_000_000)
	start = time.Now()
	s := int64(0)
	for i := int64(0); i < nArith; i++ {
		s += i * i % 7
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] loops_arith: %d iterations in %d ms (result=%d)\n", nArith, elapsed, s)

	// Part C: While loop
	nWhile := int64(500_000_000)
	start = time.Now()
	n := nWhile
	for n > 0 {
		n--
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] loops_while: %d iterations in %d ms (result=%d)\n", nWhile, elapsed, n)
}
