// Benchmark 06: Closures — lambda invoke, closure capture, higher-order functions
// Measures closure call overhead, captured variable access, function-as-argument
package main

import (
	"fmt"
	"time"
)

func apply(f func(int64) int64, x int64) int64 {
	return f(x)
}

func main() {
	// Part A: Lambda invocation
	double := func(x int64) int64 { return x * 2 }
	nInvoke := int64(200_000_000)
	start := time.Now()
	x := int64(0)
	for i := int64(0); i < nInvoke; i++ {
		x = double(i)
	}
	elapsed := time.Since(start).Milliseconds()
	fmt.Printf("[bench] closure_invoke: %d calls in %d ms (result=%d)\n", nInvoke, elapsed, x)

	// Part B: Closure capture
	offset := int64(42)
	adder := func(x int64) int64 { return x + offset }
	nCapture := int64(200_000_000)
	start = time.Now()
	x = 0
	for i := int64(0); i < nCapture; i++ {
		x = adder(i)
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] closure_capture: %d calls in %d ms (result=%d)\n", nCapture, elapsed, x)

	// Part C: Higher-order function
	inc := func(x int64) int64 { return x + 1 }
	nHof := int64(100_000_000)
	start = time.Now()
	x = 0
	for i := int64(0); i < nHof; i++ {
		x = apply(inc, i)
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] closure_hof: %d calls in %d ms (result=%d)\n", nHof, elapsed, x)
}
