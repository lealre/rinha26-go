//go:build amd64

#include "textflag.h"

// func ScanBlock8AVX2(q *float32, block *int16, worst float32, sum *[8]float32) bool
//
// Compute 8 squared L2 distances in parallel between a 14-dim float32 query
// (passed in scaled int16 coordinates) and 8 int16 reference vectors stored
// dim-major in a 224-byte block.
//
// The contract: distances are written to *sum, the return value is false
// when all 8 distances are guaranteed to exceed `worst` (block-level early
// exit), in which case the caller may skip reading sum.
//
// Block memory layout:
//   bytes [d*16, d*16+16) = the 8 int16 values for dim d (one per slot)
//
// We accumulate squared diffs into four parallel chains, striped by
// (dim mod 4). On Haswell FMA has 5-cycle latency on a single port, so a
// single-chain serialization would be 14*5 = 70 cycles end-to-end. With
// four interleaved chains the longest chain holds 4 FMAs (dims 0,4,8,12 or
// 1,5,9,13), so the FMA-bound critical path is 4*5 = 20 cycles. The
// horizontal merge at the end is three VADDPS (≈3 cycles each).
//
//   Y0  ← dims 0, 4, 8, 12  (4 FMAs)
//   Y1  ← dims 1, 5, 9, 13  (4 FMAs)
//   Y2  ← dims 2, 6, 10     (3 FMAs)
//   Y3  ← dims 3, 7, 11     (3 FMAs)
//
// Register allocation:
//   DI  = &q[0]
//   SI  = &block[0]
//   DX  = &sum[0]
//   AX  = alive-mask scratch
//   Y0..Y3 = squared-distance accumulators
//   Y4  = scratch (loaded ref dim, after int32→f32 conversion)
//   Y5  = scratch (broadcasted query dim, then diff)
//   Y6  = scratch (partial merge)
//   Y7  = scratch (cmp mask result)
//   Y15 = broadcast worst across 8 lanes (constant for the call)
//
// Stack frame: 33 bytes per the Linux/amd64 calling convention used by Go's
// assembler for this kind of stub.
//   q:     +0  (8B pointer)
//   block: +8  (8B pointer)
//   worst: +16 (4B float32, then 4B alignment pad)
//   sum:   +24 (8B pointer)
//   ret:   +32 (1B bool)

TEXT ·ScanBlock8AVX2(SB), NOSPLIT, $0-33
	MOVQ         q+0(FP), DI
	MOVQ         block+8(FP), SI
	VBROADCASTSS worst+16(FP), Y15
	MOVQ         sum+24(FP), DX

	// Prefetch the next block into L1d while we work on the current one.
	// Three cache lines starting just past this block cover the next
	// block's bytes [0..192) — the tail line is already partially warm
	// from this block's last load.
	PREFETCHT0 224(SI)
	PREFETCHT0 288(SI)
	PREFETCHT0 352(SI)

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	// --- dims 0..3 (one round across all four accumulators) -------------
	VPMOVSXWD    0(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 0(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y0

	VPMOVSXWD    16(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 4(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y1

	VPMOVSXWD    32(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 8(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y2

	VPMOVSXWD    48(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 12(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y3

	// --- dims 4..7 (second round) ----------------------------------------
	VPMOVSXWD    64(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 16(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y0

	VPMOVSXWD    80(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 20(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y1

	VPMOVSXWD    96(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 24(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y2

	VPMOVSXWD    112(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 28(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y3

	// --- mid-block early-exit gate ---------------------------------------
	// Sum all four accumulators and compare against the worst threshold.
	// If no lane is below worst the remaining 6 dims would only push
	// distances higher, so we can short-circuit.
	VADDPS    Y0, Y1, Y6
	VADDPS    Y2, Y3, Y7
	VADDPS    Y6, Y7, Y6
	VCMPPS    $0x01, Y15, Y6, Y7   // 0x01 = LT_OS — mask of (cum < worst)
	VMOVMSKPS Y7, AX
	TESTL     AX, AX
	JZ        all_dead

	// --- dims 8..11 (third round) ----------------------------------------
	VPMOVSXWD    128(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 32(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y0

	VPMOVSXWD    144(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 36(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y1

	VPMOVSXWD    160(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 40(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y2

	VPMOVSXWD    176(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 44(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y3

	// --- dims 12..13 (partial fourth round; Y2 and Y3 stay at 3 FMAs) ----
	VPMOVSXWD    192(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 48(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y0

	VPMOVSXWD    208(SI), Y4
	VCVTDQ2PS    Y4, Y4
	VBROADCASTSS 52(DI), Y5
	VSUBPS       Y4, Y5, Y5
	VFMADD231PS  Y5, Y5, Y1

	// --- final merge + liveness check ------------------------------------
	VADDPS    Y0, Y1, Y6
	VADDPS    Y2, Y3, Y7
	VADDPS    Y6, Y7, Y0       // Y0 now holds the 8 final sums
	VCMPPS    $0x01, Y15, Y0, Y6
	VMOVMSKPS Y6, AX
	TESTL     AX, AX
	JZ        all_dead

	VMOVUPS Y0, (DX)
	MOVB    $1, ret+32(FP)
	VZEROUPPER
	RET

all_dead:
	// We took the early-exit branch at the mid-block checkpoint. The
	// caller treats sum as undefined on the !alive path, but we still
	// write a consistent value so a future caller that reads sum after a
	// false return sees a well-defined merged total of however many dims
	// were accumulated so far.
	VADDPS  Y0, Y1, Y6
	VADDPS  Y2, Y3, Y7
	VADDPS  Y6, Y7, Y0
	VMOVUPS Y0, (DX)
	MOVB    $0, ret+32(FP)
	VZEROUPPER
	RET
