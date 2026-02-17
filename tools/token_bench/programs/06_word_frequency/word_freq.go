package main

import (
	"fmt"
	"strings"
)

func main() {
	text := "the quick brown fox jumps over the lazy dog the fox the dog and the cat sat on the mat the quick brown fox"

	words := strings.Fields(text)
	fmt.Printf("Total: %d\n", len(words))

	unique := make(map[string]bool)
	for _, w := range words {
		unique[w] = true
	}
	fmt.Printf("Unique: %d\n", len(unique))

	counts := make(map[string]int)
	for _, w := range words {
		counts[w]++
	}
	for word, count := range counts {
		fmt.Printf("%s: %d\n", word, count)
	}

	longest := words[0]
	for _, w := range words {
		if len(w) > len(longest) {
			longest = w
		}
	}
	fmt.Printf("Longest: %s (%d chars)\n", longest, len(longest))

	totalLen := 0
	for _, w := range words {
		totalLen += len(w)
	}
	avg := float64(totalLen) / float64(len(words))
	fmt.Printf("Avg length: %f\n", avg)
}
