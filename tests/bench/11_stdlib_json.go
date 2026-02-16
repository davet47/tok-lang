// Benchmark 11: Stdlib JSON — parse + stringify
// Measures JSON serialization/deserialization performance

package main

import (
	"encoding/json"
	"fmt"
	"time"
)

func main() {
	// Part A: Parse simple array repeatedly
	nParse := 1_000_000
	start := time.Now()
	var last interface{}
	jsonArr := []byte("[1, 2, 3, 4, 5]")
	for i := 0; i < nParse; i++ {
		json.Unmarshal(jsonArr, &last)
	}
	arr, _ := last.([]interface{})
	elapsed := time.Since(start).Milliseconds()
	fmt.Printf("[bench] json_parse_array: %d calls in %d ms (len=%d)\n", nParse, elapsed, len(arr))

	// Part B: Stringify map repeatedly
	nStringify := 1_000_000
	data := map[string]interface{}{
		"name":    "tok",
		"version": 1,
		"fast":    true,
		"tags":    []string{"lang", "compiled"},
	}
	start = time.Now()
	var lastS []byte
	for i := 0; i < nStringify; i++ {
		lastS, _ = json.Marshal(data)
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] json_stringify: %d calls in %d ms (len=%d)\n", nStringify, elapsed, len(lastS))

	// Part C: Roundtrip parse + stringify
	nRoundtrip := 500_000
	jsonStr := []byte(`{"x":1,"y":2,"z":3}`)
	start = time.Now()
	var lastRT []byte
	for i := 0; i < nRoundtrip; i++ {
		var obj interface{}
		json.Unmarshal(jsonStr, &obj)
		lastRT, _ = json.Marshal(obj)
	}
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("[bench] json_roundtrip: %d parse+stringify in %d ms (last=%s)\n", nRoundtrip, elapsed, string(lastRT))
}
