package main

import (
	"fmt"
	"sort"
)

type Record struct {
	Name     string
	Score    int
	Category string
}

func lcg(seed int) int {
	return (seed*1103515245 + 12345) % 2147483648
}

func main() {
	records := make([]Record, 0, 5000)
	seed := 42
	for i := 0; i < 5000; i++ {
		seed = lcg(seed)
		score := seed % 100
		cat := score % 4
		category := "delta"
		switch cat {
		case 0:
			category = "alpha"
		case 1:
			category = "beta"
		case 2:
			category = "gamma"
		}
		records = append(records, Record{Name: fmt.Sprintf("item%d", i), Score: score, Category: category})
	}

	var highScores []Record
	for _, r := range records {
		if r.Score >= 50 {
			highScores = append(highScores, r)
		}
	}

	transformed := make([]Record, len(highScores))
	for i, r := range highScores {
		transformed[i] = Record{Name: r.Name, Score: r.Score * 2, Category: r.Category}
	}

	groups := make(map[string][]int)
	for _, r := range transformed {
		groups[r.Category] = append(groups[r.Category], r.Score)
	}

	for k, scores := range groups {
		sort.Ints(scores)
		total := 0
		for _, s := range scores {
			total += s
		}
		avg := total / len(scores)
		fmt.Printf("%s: count=%d avg=%d min=%d max=%d\n", k, len(scores), avg, scores[0], scores[len(scores)-1])
	}
}
