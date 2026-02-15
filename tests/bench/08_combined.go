// Benchmark 08: Combined — realistic data processing pipeline
// Exercises: loops, arrays, maps, closures, filter, reduce, strings
package main

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

func lcg(seed int64) int64 {
	return (seed*1103515245 + 12345) % 2147483648
}

type Record struct {
	Name     string
	Score    int64
	Category string
}

type Stats struct {
	Count int
	Avg   int64
	Min   int64
	Max   int64
}

func main() {
	nRecords := 500_000

	// Build dataset
	buildStart := time.Now()
	records := make([]Record, 0, nRecords)
	seed := int64(42)
	for i := 0; i < nRecords; i++ {
		seed = lcg(seed)
		score := seed % 100
		cat := score % 4
		var category string
		switch cat {
		case 0:
			category = "alpha"
		case 1:
			category = "beta"
		case 2:
			category = "gamma"
		default:
			category = "delta"
		}
		records = append(records, Record{
			Name:     fmt.Sprintf("item%d", i),
			Score:    score,
			Category: category,
		})
	}
	buildMs := time.Since(buildStart).Milliseconds()

	// Filter: keep records with score >= 50
	filtStart := time.Now()
	highScores := make([]Record, 0)
	for _, r := range records {
		if r.Score >= 50 {
			highScores = append(highScores, r)
		}
	}
	filtMs := time.Since(filtStart).Milliseconds()

	// Transform: double scores
	transStart := time.Now()
	transformed := make([]Record, len(highScores))
	for i, r := range highScores {
		transformed[i] = Record{
			Name:     r.Name,
			Score:    r.Score * 2,
			Category: r.Category,
		}
	}
	transMs := time.Since(transStart).Milliseconds()

	// Group by category
	groupStart := time.Now()
	groups := make(map[string][]int64)
	for _, r := range transformed {
		groups[r.Category] = append(groups[r.Category], r.Score)
	}
	groupMs := time.Since(groupStart).Milliseconds()

	// Sort each group and compute stats
	sortStart := time.Now()
	stats := make(map[string]Stats)
	for k, scores := range groups {
		sort.Slice(scores, func(i, j int) bool { return scores[i] < scores[j] })
		count := len(scores)
		sum := int64(0)
		for _, s := range scores {
			sum += s
		}
		avg := sum / int64(count)
		stats[k] = Stats{
			Count: count,
			Avg:   avg,
			Min:   scores[0],
			Max:   scores[count-1],
		}
	}
	sortMs := time.Since(sortStart).Milliseconds()

	// Format output
	fmtStart := time.Now()
	var sb strings.Builder
	statKeys := make([]string, 0, len(stats))
	for k := range stats {
		statKeys = append(statKeys, k)
	}
	sort.Strings(statKeys)
	for _, k := range statKeys {
		s := stats[k]
		sb.WriteString(fmt.Sprintf("%s: count=%d avg=%d min=%d max=%d\n", k, s.Count, s.Avg, s.Min, s.Max))
	}
	output := sb.String()
	fmtMs := time.Since(fmtStart).Milliseconds()

	totalMs := buildMs + filtMs + transMs + groupMs + sortMs + fmtMs
	fmt.Printf("[bench] combined: %d records processed in %d ms\n", nRecords, totalMs)
	fmt.Printf("[bench] combined_detail: build=%dms filter=%dms transform=%dms group=%dms sort=%dms format=%dms\n",
		buildMs, filtMs, transMs, groupMs, sortMs, fmtMs)
	fmt.Printf("[bench] combined_counts: total=%d filtered=%d groups=%d\n",
		len(records), len(transformed), len(statKeys))
	fmt.Print(output)
}
