// Benchmark 09: Stdlib Math — trigonometry, sqrt, pow, log
// Measures math function call overhead

package main

import (
	"fmt"
	"math"
	"time"
)

func main() {
	// Part A: sqrt loop
	nSqrt := 10_000_000
	start := time.Now()
	total := 0.0
	for i := 1; i <= nSqrt; i++ {
		total += math.Sqrt(float64(i))
	}
	elapsed := time.Since(start).Milliseconds()
	fmt.Printf("[bench] math_sqrt: %d calls in %d ms (result=%d)\n", nSqrt, elapsed, int64(math.Floor(total)))

	// Part B: sin + cos
	nTrig := 5_000_000
	start = time.Now()
	sum := 0.0
	for i := 0; i < nTrig; i++ {
		x := float64(i) * 0.001
		sum += math.Sin(x) + math.Cos(x)
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] math_trig: %d sin+cos pairs in %d ms (result=%d)\n", nTrig, elapsed, int64(math.Floor(sum)))

	// Part C: pow + log roundtrip
	nPowlog := 5_000_000
	start = time.Now()
	acc := 0.0
	for i := 1; i <= nPowlog; i++ {
		acc += math.Log(math.Pow(2.0, float64(i%20)) + 1.0)
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] math_powlog: %d pow+log calls in %d ms (result=%d)\n", nPowlog, elapsed, int64(math.Floor(acc)))
}
