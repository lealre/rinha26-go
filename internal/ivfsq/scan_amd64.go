//go:build amd64

package ivfsq

// ScanBlock8AVX2 computes 8 squared L2 distances between the 14-dim float32
// query q and 8 int16 vectors stored dim-major in block (14*8 int16 = 224
// bytes). Distances are written to *sum. The function returns false if all
// 8 vectors are guaranteed not to make top-5 against the worst threshold
// (early-exit), in which case the caller can skip reading sum.
//
// Block layout: block[d*8 + s] holds the s-th vector's d-th dim.
//
// Implementation: AVX2 + FMA3 hand-written assembly with two parallel FMA
// accumulator chains (even/odd dims) for halved critical path. The kernel
// itself emits VFMADD231PS / VPMOVSXWD / VCVTDQ2PS etc. regardless of
// GOAMD64 — required CPU features are AVX2 + FMA3 (Haswell, 2013+).
//
//go:noescape
func ScanBlock8AVX2(q *float32, block *int16, worst float32, sum *[8]float32) bool
