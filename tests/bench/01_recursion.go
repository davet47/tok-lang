// Benchmark 01: Recursion — naive recursive Fibonacci
// Measures function call overhead, stack frame creation
package main

import (
	"fmt"
	"time"
)

func fib(n int) int {
	if n < 2 {
		return n
	}
	return fib(n-1) + fib(n-2)
}

func main() {
	n := 45
	start := time.Now()
	r := fib(n)
	elapsed := time.Since(start).Milliseconds()
	fmt.Printf("[bench] recursion: fib(%d) = %d in %d ms\n", n, r, elapsed)
}
