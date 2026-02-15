// Benchmark 07: Concurrency — goroutines, channels, parallel map
// Measures goroutine spawn/join, channel throughput, parallel speedup
package main

import (
	"fmt"
	"sync"
	"time"
)

func fib(n int) int {
	if n < 2 {
		return n
	}
	return fib(n-1) + fib(n-2)
}

func main() {
	// Part A: Parallel goroutines (8x fib(27))
	start := time.Now()
	results := make([]int, 8)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = fib(27)
		}(i)
	}
	wg.Wait()
	r := 0
	for _, v := range results {
		r += v
	}
	parMs := time.Since(start).Milliseconds()
	fmt.Printf("[bench] concurrency_parallel: 8 goroutines fib(27) in %d ms (result=%d)\n", parMs, r)

	// Part B: Channel throughput
	nMsgs := 10_000_000
	ch := make(chan int64, 100)
	start = time.Now()
	go func() {
		for i := int64(0); i < int64(nMsgs); i++ {
			ch <- i
		}
	}()
	total := int64(0)
	for i := 0; i < nMsgs; i++ {
		total += <-ch
	}
	elapsed := time.Since(start).Milliseconds()
	fmt.Printf("[bench] concurrency_channel: %d messages in %d ms (sum=%d)\n", nMsgs, elapsed, total)

	// Part C: Parallel map (8x fib(27))
	start = time.Now()
	pmapResults := make([]int, 8)
	var wg2 sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg2.Add(1)
		go func(idx int) {
			defer wg2.Done()
			pmapResults[idx] = fib(27)
		}(i)
	}
	wg2.Wait()
	r2 := 0
	for _, v := range pmapResults {
		r2 += v
	}
	pmapMs := time.Since(start).Milliseconds()
	fmt.Printf("[bench] concurrency_pmap: 8 elements in %d ms (result=%d)\n", pmapMs, r2)

	// Part D: Sequential comparison
	start = time.Now()
	total2 := 0
	for i := 0; i < 8; i++ {
		total2 += fib(27)
	}
	seqMs := time.Since(start).Milliseconds()
	fmt.Printf("[bench] concurrency_sequential: 8 calls in %d ms (result=%d)\n", seqMs, total2)

	// Speedup ratio
	speedup := float64(seqMs) / float64(parMs)
	fmt.Printf("[bench] concurrency_speedup: %.1fx (parallel vs sequential)\n", speedup)
}
