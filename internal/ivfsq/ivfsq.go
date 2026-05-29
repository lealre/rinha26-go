// Package ivfsq is an IVF + int16-scalar-quantized vector index with an
// AVX2 distance kernel.
//
// Storage layout v02: each cluster's vectors are organized into "blocks" of
// 8 vectors, stored dim-major within each block:
//
//	block[d*8 + s]  = dim d of the s-th vector in the block
//
// One block is 14 × 8 = 112 int16 = 224 bytes — exactly the input format
// expected by ScanBlock8AVX2 in scan_amd64.s. The last block of each cluster
// is padded with sentinel int16 values so partial blocks always have 8 slots
// to read; padded slots produce huge distances and never enter top-5.
//
// At query time, the leaf distance loop walks the blocks of each probed
// cluster and calls ScanBlock8AVX2(q, block, worst, sum). The kernel handles
// the per-block early-exit gate internally (it bails after dim 8 if all 8
// vectors are already past the threshold).
package ivfsq

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"unsafe"
)

// IVFSQ tunables. Bump the magic version on any layout change.
const (
	IVFSQ_K       = 1024 // number of IVF clusters
	IVFSQ_Dim     = 14   // vector dimensionality
	IVFSQ_TopK    = 5    // number of nearest neighbors to retrieve
	IVFSQ_NProbe  = 8    // clusters probed per query
	IVFSQ_Scale   = 32767
	BlockVecCount = 8                                  // vectors per block
	BlockInts     = IVFSQ_Dim * BlockVecCount          // int16 per block (= 112)
	BlockBytes    = BlockInts * 2                      // bytes per block (= 224)
	SentinelI16   = 32767                              // padding value: produces huge distance, never wins top-5
)

// IVFSQ_Magic is the 8-byte file-format identifier written to /index.bin.
// v02 introduced the dim-major-block-of-8 leaf layout for the AVX2 kernel.
var IVFSQ_Magic = [8]byte{'I', 'V', 'F', 'S', 'Q', 'v', '0', '2'}

// IVFSQ is the in-memory index. All slices alias the file buffer returned by
// os.ReadFile (zero-copy on amd64).
type IVFSQ struct {
	K           uint32 // = IVFSQ_K
	N           uint32 // total number of real vectors (excludes padding)
	TotalBlocks uint32 // total number of blocks across all clusters
	IVFCentroids []float32 // K * Dim float32
	BlockOffsets []uint32  // K+1 — cumulative block counts, NOT int16 offsets
	BlockData    []int16   // TotalBlocks * BlockInts (dim-major within block)
	BlockLabels  []uint8   // TotalBlocks * BlockVecCount (1 = fraud, 0 = legit / padding)
}

// quantize maps a normalized float32 to int16 in [-32767, 32767].
func quantize(v float32) int16 {
	x := math.Round(float64(v) * float64(IVFSQ_Scale))
	if x > float64(IVFSQ_Scale) {
		x = float64(IVFSQ_Scale)
	} else if x < -float64(IVFSQ_Scale) {
		x = -float64(IVFSQ_Scale)
	}
	return int16(x)
}

// Save writes the index to path in the v02 binary layout.
func (ivf *IVFSQ) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)

	if _, err := w.Write(IVFSQ_Magic[:]); err != nil {
		return fmt.Errorf("write magic: %w", err)
	}
	hdr := []uint32{ivf.K, ivf.N, ivf.TotalBlocks}
	if err := binary.Write(w, binary.LittleEndian, hdr); err != nil {
		return fmt.Errorf("write header: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, ivf.IVFCentroids); err != nil {
		return fmt.Errorf("write IVFCentroids: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, ivf.BlockOffsets); err != nil {
		return fmt.Errorf("write BlockOffsets: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, ivf.BlockData); err != nil {
		return fmt.Errorf("write BlockData: %w", err)
	}
	if _, err := w.Write(ivf.BlockLabels); err != nil {
		return fmt.Errorf("write BlockLabels: %w", err)
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush: %w", err)
	}
	return nil
}

// LoadIndex reads a v02 index from path. Slices alias the file buffer; the
// underlying byte slice must stay reachable for the lifetime of the index.
// Assumes little-endian (amd64).
func LoadIndex(path string) (*IVFSQ, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	const headerSize = 8 + 4*3 // magic + K + N + TotalBlocks
	if len(data) < headerSize {
		return nil, fmt.Errorf("file %s too small: %d bytes", path, len(data))
	}

	var magic [8]byte
	copy(magic[:], data[0:8])
	if magic != IVFSQ_Magic {
		return nil, fmt.Errorf("bad magic in %s: got %q, want %q",
			path, magic[:], IVFSQ_Magic[:])
	}

	ivf := &IVFSQ{
		K:           binary.LittleEndian.Uint32(data[8:12]),
		N:           binary.LittleEndian.Uint32(data[12:16]),
		TotalBlocks: binary.LittleEndian.Uint32(data[16:20]),
	}
	if ivf.K != IVFSQ_K {
		return nil, fmt.Errorf("unsupported K in %s: %d (want %d)", path, ivf.K, IVFSQ_K)
	}

	offset := headerSize

	centroidsBytes := int(ivf.K) * IVFSQ_Dim * 4
	if offset+centroidsBytes > len(data) {
		return nil, fmt.Errorf("file truncated at IVFCentroids")
	}
	ivf.IVFCentroids = bytesToFloat32(data[offset : offset+centroidsBytes])
	offset += centroidsBytes

	offsetsBytes := (int(ivf.K) + 1) * 4
	if offset+offsetsBytes > len(data) {
		return nil, fmt.Errorf("file truncated at BlockOffsets")
	}
	ivf.BlockOffsets = bytesToUint32(data[offset : offset+offsetsBytes])
	offset += offsetsBytes

	blockBytes := int(ivf.TotalBlocks) * BlockBytes
	if offset+blockBytes > len(data) {
		return nil, fmt.Errorf("file truncated at BlockData")
	}
	ivf.BlockData = bytesToInt16(data[offset : offset+blockBytes])
	offset += blockBytes

	labelsBytes := int(ivf.TotalBlocks) * BlockVecCount
	if offset+labelsBytes > len(data) {
		return nil, fmt.Errorf("file truncated at BlockLabels")
	}
	ivf.BlockLabels = data[offset : offset+labelsBytes]
	offset += labelsBytes

	if offset != len(data) {
		return nil, fmt.Errorf("trailing %d bytes in %s", len(data)-offset, path)
	}
	return ivf, nil
}

// Build trains the IVFSQ index. Called only by cmd/build-index (offline).
//
// vectors: flat slice of N*Dim float32 (row-major per vector).
// labels: N uint8 (1 = fraud, 0 = legit).
// seed: deterministic across rebuilds when fixed.
func Build(vectors []float32, labels []uint8, seed int64) *IVFSQ {
	n := len(labels)
	if len(vectors) != n*IVFSQ_Dim {
		log.Fatalf("Build: vectors length %d does not match N*Dim = %d*%d = %d",
			len(vectors), n, IVFSQ_Dim, n*IVFSQ_Dim)
	}

	log.Printf("Build: %d vectors", n)
	log.Printf("Build: training IVF centroids (K=%d, dim=%d)...", IVFSQ_K, IVFSQ_Dim)
	ivfRes := KMeans(vectors, n, IVFSQ_Dim, IVFSQ_K, 30, seed)
	log.Printf("Build: IVF centroids done")

	// Count vectors per cluster, then compute block counts.
	counts := make([]uint32, IVFSQ_K)
	for _, c := range ivfRes.Assignments {
		counts[c]++
	}
	blockCounts := make([]uint32, IVFSQ_K)
	totalBlocks := uint32(0)
	for c := 0; c < IVFSQ_K; c++ {
		blockCounts[c] = (counts[c] + BlockVecCount - 1) / BlockVecCount
		totalBlocks += blockCounts[c]
	}
	blockOffsets := make([]uint32, IVFSQ_K+1)
	for c := 0; c < IVFSQ_K; c++ {
		blockOffsets[c+1] = blockOffsets[c] + blockCounts[c]
	}

	// Reverse map: each cluster c stores []vectorIdx of length counts[c].
	// vectorOrder[c] = []int of source indices in cluster c, in their
	// original encounter order.
	clusterMembers := make([][]int, IVFSQ_K)
	for c := 0; c < IVFSQ_K; c++ {
		clusterMembers[c] = make([]int, 0, counts[c])
	}
	for i := 0; i < n; i++ {
		c := ivfRes.Assignments[i]
		clusterMembers[c] = append(clusterMembers[c], i)
	}

	blockData := make([]int16, int(totalBlocks)*BlockInts)
	blockLabels := make([]uint8, int(totalBlocks)*BlockVecCount)

	// Pre-fill all slots with sentinel so partial-final-block padding works.
	for i := range blockData {
		blockData[i] = SentinelI16
	}

	log.Printf("Build: laying out %d blocks (dim-major-of-8)...", totalBlocks)
	for c := 0; c < IVFSQ_K; c++ {
		members := clusterMembers[c]
		clusterBlockStart := blockOffsets[c]
		for b := uint32(0); b < blockCounts[c]; b++ {
			globalBlock := int(clusterBlockStart + b)
			blockDataOff := globalBlock * BlockInts
			blockLabelOff := globalBlock * BlockVecCount
			for s := 0; s < BlockVecCount; s++ {
				localIdx := int(b)*BlockVecCount + s
				if localIdx >= len(members) {
					// Padding slot: keep SentinelI16 across all dims; label 0.
					continue
				}
				vIdx := members[localIdx]
				for d := 0; d < IVFSQ_Dim; d++ {
					blockData[blockDataOff+d*BlockVecCount+s] = quantize(vectors[vIdx*IVFSQ_Dim+d])
				}
				blockLabels[blockLabelOff+s] = labels[vIdx]
			}
		}
	}
	log.Printf("Build: done; total_blocks=%d, mean_vecs_per_block=%.2f",
		totalBlocks, float64(n)/float64(totalBlocks))

	return &IVFSQ{
		K:            IVFSQ_K,
		N:            uint32(n),
		TotalBlocks:  totalBlocks,
		IVFCentroids: ivfRes.Centroids,
		BlockOffsets: blockOffsets,
		BlockData:    blockData,
		BlockLabels:  blockLabels,
	}
}

// Search returns the count of fraud labels among the IVFSQ_TopK nearest
// reference vectors to the query.
//
// Path:
//  1. Scalar scan of K=1024 centroids → top NProbe clusters.
//  2. For each probed cluster, walk its blocks and call ScanBlock8AVX2 to
//     get 8 squared distances per call. Insert any below-threshold winners
//     into the top-5 buffer.
//  3. Tally fraud labels among the final top-5.
func (ivf *IVFSQ) Search(query [IVFSQ_Dim]float32) int {
	// 1. Find the NProbe nearest IVF centroids (float32 distance, scalar).
	type centDist struct {
		idx  uint32
		dist float32
	}
	var topProbes [IVFSQ_NProbe]centDist
	for i := range topProbes {
		topProbes[i].dist = float32(1e30)
	}
	for c := uint32(0); c < ivf.K; c++ {
		var d float32
		base := int(c) * IVFSQ_Dim
		for j := 0; j < IVFSQ_Dim; j++ {
			diff := query[j] - ivf.IVFCentroids[base+j]
			d += diff * diff
		}
		if d >= topProbes[IVFSQ_NProbe-1].dist {
			continue
		}
		k := IVFSQ_NProbe - 1
		for k > 0 && topProbes[k-1].dist > d {
			topProbes[k] = topProbes[k-1]
			k--
		}
		topProbes[k] = centDist{idx: c, dist: d}
	}

	// 2. Walk blocks per probed cluster with the AVX2 kernel.
	type cand struct {
		dist  float32
		fraud bool
	}
	var top [IVFSQ_TopK]cand
	for i := range top {
		top[i].dist = math.MaxFloat32
	}
	threshold := top[IVFSQ_TopK-1].dist

	// The kernel expects an int16 query in scaled units to match the
	// quantized refs. We pass float32 query in *scaled* units (multiplied by
	// IVFSQ_Scale) so the kernel's VCVTDQ2PS(ref) and broadcast(query) live
	// in the same coordinate system.
	var qScaled [IVFSQ_Dim]float32
	for j := 0; j < IVFSQ_Dim; j++ {
		qScaled[j] = query[j] * float32(IVFSQ_Scale)
	}

	var sum [8]float32

	for _, probe := range topProbes {
		c := probe.idx
		blockStart := ivf.BlockOffsets[c]
		blockEnd := ivf.BlockOffsets[c+1]
		for b := blockStart; b < blockEnd; b++ {
			blockPtr := &ivf.BlockData[int(b)*BlockInts]
			alive := ScanBlock8AVX2(&qScaled[0], blockPtr, threshold, &sum)
			if !alive {
				continue
			}
			labelOff := int(b) * BlockVecCount
			for s := 0; s < BlockVecCount; s++ {
				if sum[s] >= threshold {
					continue
				}
				fraud := ivf.BlockLabels[labelOff+s] == 1
				k := IVFSQ_TopK - 1
				for k > 0 && top[k-1].dist > sum[s] {
					top[k] = top[k-1]
					k--
				}
				top[k] = cand{dist: sum[s], fraud: fraud}
				threshold = top[IVFSQ_TopK-1].dist
			}
		}
	}

	count := 0
	for _, e := range top {
		if e.fraud {
			count++
		}
	}
	return count
}

// bytesToFloat32 reinterprets a byte slice as float32 (zero-copy, amd64).
func bytesToFloat32(b []byte) []float32 {
	if len(b)%4 != 0 {
		panic("bytesToFloat32: length not multiple of 4")
	}
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// bytesToUint32 reinterprets a byte slice as uint32 (zero-copy, amd64).
func bytesToUint32(b []byte) []uint32 {
	if len(b)%4 != 0 {
		panic("bytesToUint32: length not multiple of 4")
	}
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// bytesToInt16 reinterprets a byte slice as int16 (zero-copy, amd64).
func bytesToInt16(b []byte) []int16 {
	if len(b)%2 != 0 {
		panic("bytesToInt16: length not multiple of 2")
	}
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*int16)(unsafe.Pointer(&b[0])), len(b)/2)
}
