package main

import "fmt"

func qsort(a []int) []int {
	if len(a) <= 1 {
		return a
	}
	p := a[len(a)/2]
	var lo, eq, hi []int
	for _, x := range a {
		if x < p {
			lo = append(lo, x)
		} else if x == p {
			eq = append(eq, x)
		} else {
			hi = append(hi, x)
		}
	}
	result := qsort(lo)
	result = append(result, eq...)
	result = append(result, qsort(hi)...)
	return result
}

func main() {
	fmt.Println(qsort([]int{3, 6, 8, 10, 1, 2, 1}))
}
