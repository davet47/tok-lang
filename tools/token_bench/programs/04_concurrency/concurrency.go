package main

import (
	"fmt"
	"sync"
)

func fib(n int) int {
	if n < 2 {
		return n
	}
	return fib(n-1) + fib(n-2)
}

func main() {
	results := make([]int, 4)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
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
	fmt.Printf("parallel: %d\n", r)

	results2 := make([]int, 4)
	var wg2 sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg2.Add(1)
		go func(idx int) {
			defer wg2.Done()
			results2[idx] = fib(27)
		}(i)
	}
	wg2.Wait()
	total := 0
	for _, v := range results2 {
		total += v
	}
	fmt.Printf("pmap: %d\n", total)

	ch := make(chan int, 10)
	go func() {
		for i := 0; i < 10; i++ {
			ch <- i * i
		}
	}()
	for i := 0; i < 10; i++ {
		fmt.Println(<-ch)
	}
}
