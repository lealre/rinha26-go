package main

import "math"

// topKFraudCount returns the number of fraud labels among the 5 reference
// vectors with smallest squared Euclidean distance to query.
//
// Distance is squared Euclidean (no sqrt) — ordering is identical and we
// skip ~3M square roots per request. The top-5 is maintained as a fixed
// 5-element sorted array; insertion is O(K) shifts, cheaper than a heap
// for K=5.
//
// Tie-break: when two reference distances are exactly equal, the one
// encountered first wins (strict > in the shift loop).
func topKFraudCount(query [14]float32, vectors [][14]float32, labels []bool) int {
	type entry struct {
		dist  float32
		fraud bool
	}

	var top [5]entry
	for i := range top {
		top[i].dist = float32(math.Inf(1))
	}

	for i, v := range vectors {
		var d float32
		for j := 0; j < 14; j++ {
			diff := query[j] - v[j]
			d += diff * diff
		}
		if d >= top[4].dist {
			continue
		}
		// shift entries with dist > d one slot right, then insert at the hole
		k := 4
		for k > 0 && top[k-1].dist > d {
			top[k] = top[k-1]
			k--
		}
		top[k] = entry{dist: d, fraud: labels[i]}
	}

	count := 0
	for _, e := range top {
		if e.fraud {
			count++
		}
	}
	return count
}
