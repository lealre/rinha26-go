//go:build !amd64

package ivfsq

import "unsafe"

// ScanBlock8AVX2 fallback for non-amd64: scalar implementation matching the
// same contract as the AVX2 kernel. Same block layout (dim-major, 14*8 int16).
// Used only for cross-arch local builds — Docker target is linux/amd64.
func ScanBlock8AVX2(q *float32, block *int16, worst float32, sum *[8]float32) bool {
	qSlice := (*[14]float32)(unsafe.Pointer(q))
	blockSlice := (*[14 * 8]int16)(unsafe.Pointer(block))

	anyAlive := false
	for s := 0; s < 8; s++ {
		var acc float32
		for d := 0; d < 14; d++ {
			diff := qSlice[d] - float32(blockSlice[d*8+s])
			acc += diff * diff
		}
		sum[s] = acc
		if acc < worst {
			anyAlive = true
		}
	}
	return anyAlive
}
