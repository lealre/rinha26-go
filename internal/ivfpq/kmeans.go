package ivfpq

import (
	"math/rand"
)

// KMeansResult holds the trained centroids and the per-point assignments.
//
// Centroids is a flat slice of K*Dim float32 values, row-major: centroid c's
// dimensions live in Centroids[c*Dim : (c+1)*Dim].
//
// Assignments is a slice of length N giving the cluster index in [0, K) for
// each input point.
type KMeansResult struct {
	Centroids   []float32
	Assignments []uint32
}

// KMeans runs Lloyd's algorithm with k-means++ initialization.
//
// data: flat slice of N*Dim float32 values, row-major.
// n:    number of points.
// dim:  dimensionality of each point.
// k:    number of clusters.
// iters: number of Lloyd iterations after init.
// seed: RNG seed (deterministic builds across rebuilds when fixed).
//
// Dead clusters (zero assignments at the end of an iteration) are re-seeded
// from a random data point.
func KMeans(data []float32, n, dim, k, iters int, seed int64) KMeansResult {
	rng := rand.New(rand.NewSource(seed))
	centroids := initKMeansPP(data, n, dim, k, rng)
	assignments := make([]uint32, n)

	for iter := 0; iter < iters; iter++ {
		// E-step: assign each point to its nearest centroid.
		for i := 0; i < n; i++ {
			assignments[i] = nearest(data[i*dim:(i+1)*dim], centroids, k, dim)
		}
		// M-step: recompute centroids as the mean of assigned points.
		newCentroids := make([]float32, k*dim)
		counts := make([]int, k)
		for i := 0; i < n; i++ {
			c := assignments[i]
			for d := 0; d < dim; d++ {
				newCentroids[int(c)*dim+d] += data[i*dim+d]
			}
			counts[c]++
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				// Dead cluster: re-seed from a random point.
				idx := rng.Intn(n)
				copy(newCentroids[c*dim:(c+1)*dim], data[idx*dim:(idx+1)*dim])
				continue
			}
			inv := 1.0 / float32(counts[c])
			for d := 0; d < dim; d++ {
				newCentroids[c*dim+d] *= inv
			}
		}
		centroids = newCentroids
	}

	// Final assignment pass with the converged centroids.
	for i := 0; i < n; i++ {
		assignments[i] = nearest(data[i*dim:(i+1)*dim], centroids, k, dim)
	}
	return KMeansResult{Centroids: centroids, Assignments: assignments}
}

// initKMeansPP picks K initial centroids using the k-means++ scheme:
// first centroid uniform-random, each next chosen with probability
// proportional to its squared distance to the nearest existing centroid.
func initKMeansPP(data []float32, n, dim, k int, rng *rand.Rand) []float32 {
	centroids := make([]float32, k*dim)

	// First centroid: uniform random point.
	first := rng.Intn(n)
	copy(centroids[0:dim], data[first*dim:(first+1)*dim])

	// minSqDist[i] = squared distance from point i to the closest centroid so far.
	minSqDist := make([]float32, n)
	for i := 0; i < n; i++ {
		minSqDist[i] = sqDist(data[i*dim:(i+1)*dim], centroids[0:dim])
	}

	for c := 1; c < k; c++ {
		// Sample the next centroid index weighted by minSqDist.
		var sum float32
		for i := 0; i < n; i++ {
			sum += minSqDist[i]
		}
		r := rng.Float32() * sum
		idx := n - 1
		for i := 0; i < n; i++ {
			r -= minSqDist[i]
			if r <= 0 {
				idx = i
				break
			}
		}
		copy(centroids[c*dim:(c+1)*dim], data[idx*dim:(idx+1)*dim])

		// Update minSqDist for the new centroid.
		newCent := centroids[c*dim : (c+1)*dim]
		for i := 0; i < n; i++ {
			d := sqDist(data[i*dim:(i+1)*dim], newCent)
			if d < minSqDist[i] {
				minSqDist[i] = d
			}
		}
	}
	return centroids
}

// nearest returns the index of the centroid (in [0, K)) closest to point.
func nearest(point, centroids []float32, k, dim int) uint32 {
	bestIdx := uint32(0)
	bestDist := sqDist(point, centroids[0:dim])
	for c := 1; c < k; c++ {
		d := sqDist(point, centroids[c*dim:(c+1)*dim])
		if d < bestDist {
			bestDist = d
			bestIdx = uint32(c)
		}
	}
	return bestIdx
}

// sqDist returns the squared Euclidean distance between two equal-length vectors.
func sqDist(a, b []float32) float32 {
	var d float32
	for i := range a {
		diff := a[i] - b[i]
		d += diff * diff
	}
	return d
}
