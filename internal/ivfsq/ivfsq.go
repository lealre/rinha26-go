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

// IVFSQ parameter constants. Bump the magic version on any layout change.
const (
	IVFSQ_K      = 1024 // number of IVF clusters
	IVFSQ_Dim    = 14   // vector dimensionality
	IVFSQ_TopK   = 5    // number of nearest neighbors to retrieve
	IVFSQ_NProbe = 8    // clusters probed per query (was 16; halved to cut p99 since detection rate component is already saturated at nprobe=16)
	IVFSQ_Scale  = 32767
)

// IVFSQ_Magic is the 8-byte file-format identifier written to /index.bin.
var IVFSQ_Magic = [8]byte{'I', 'V', 'F', 'S', 'Q', 'v', '0', '1'}

// IVFSQ is the in-memory IVF + int16-scalar-quantized index.
//
// Quantized and Labels alias the underlying byte buffer returned by
// os.ReadFile; the buffer stays alive as long as any of these slices is
// reachable.
type IVFSQ struct {
	K uint32 // = IVFSQ_K
	N uint32 // total number of indexed vectors

	IVFCentroids   []float32 // K * IVFSQ_Dim
	ClusterOffsets []uint32  // K+1
	Quantized      []int16   // N * IVFSQ_Dim, grouped by cluster
	Labels         []uint8   // N (0 = legit, 1 = fraud)
}

// quantize maps a single normalized float32 value to int16 in [-32767, 32767].
func quantize(v float32) int16 {
	x := math.Round(float64(v) * float64(IVFSQ_Scale))
	if x > float64(IVFSQ_Scale) {
		x = float64(IVFSQ_Scale)
	} else if x < -float64(IVFSQ_Scale) {
		x = -float64(IVFSQ_Scale)
	}
	return int16(x)
}

// Save serializes the index to path using the binary layout in the spec.
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
	hdr := []uint32{ivf.K, ivf.N}
	if err := binary.Write(w, binary.LittleEndian, hdr); err != nil {
		return fmt.Errorf("write header: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, ivf.IVFCentroids); err != nil {
		return fmt.Errorf("write IVFCentroids: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, ivf.ClusterOffsets); err != nil {
		return fmt.Errorf("write ClusterOffsets: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, ivf.Quantized); err != nil {
		return fmt.Errorf("write Quantized: %w", err)
	}
	if _, err := w.Write(ivf.Labels); err != nil {
		return fmt.Errorf("write Labels: %w", err)
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush: %w", err)
	}
	return nil
}

// LoadIndex reads an IVFSQ index from path. Slices alias the file buffer.
// Assumes little-endian host (amd64). The challenge spec mandates amd64.
func LoadIndex(path string) (*IVFSQ, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	const headerSize = 16
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
		K: binary.LittleEndian.Uint32(data[8:12]),
		N: binary.LittleEndian.Uint32(data[12:16]),
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
		return nil, fmt.Errorf("file truncated at ClusterOffsets")
	}
	ivf.ClusterOffsets = bytesToUint32(data[offset : offset+offsetsBytes])
	offset += offsetsBytes

	quantizedBytes := int(ivf.N) * IVFSQ_Dim * 2
	if offset+quantizedBytes > len(data) {
		return nil, fmt.Errorf("file truncated at Quantized")
	}
	ivf.Quantized = bytesToInt16(data[offset : offset+quantizedBytes])
	offset += quantizedBytes

	labelsBytes := int(ivf.N)
	if offset+labelsBytes > len(data) {
		return nil, fmt.Errorf("file truncated at Labels")
	}
	ivf.Labels = data[offset : offset+labelsBytes]
	offset += labelsBytes

	if offset != len(data) {
		return nil, fmt.Errorf("trailing %d bytes in %s", len(data)-offset, path)
	}
	return ivf, nil
}

// Build trains the IVFSQ index. Called only by cmd/build-index (offline).
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

	// Count cluster sizes.
	counts := make([]uint32, IVFSQ_K)
	for _, c := range ivfRes.Assignments {
		counts[c]++
	}
	offsets := make([]uint32, IVFSQ_K+1)
	for c := 0; c < IVFSQ_K; c++ {
		offsets[c+1] = offsets[c] + counts[c]
	}
	if offsets[IVFSQ_K] != uint32(n) {
		log.Fatalf("Build: offset sum mismatch: got %d, want %d", offsets[IVFSQ_K], n)
	}

	// Fill Quantized and Labels in cluster-grouped order.
	quantized := make([]int16, n*IVFSQ_Dim)
	labs := make([]uint8, n)
	cursor := make([]uint32, IVFSQ_K)
	copy(cursor, offsets[:IVFSQ_K])

	log.Printf("Build: quantizing %d vectors to int16...", n)
	for i := 0; i < n; i++ {
		c := ivfRes.Assignments[i]
		pos := cursor[c]
		cursor[c]++
		for j := 0; j < IVFSQ_Dim; j++ {
			quantized[int(pos)*IVFSQ_Dim+j] = quantize(vectors[i*IVFSQ_Dim+j])
		}
		labs[pos] = labels[i]
	}
	log.Printf("Build: done")

	return &IVFSQ{
		K:              IVFSQ_K,
		N:              uint32(n),
		IVFCentroids:   ivfRes.Centroids,
		ClusterOffsets: offsets,
		Quantized:      quantized,
		Labels:         labs,
	}
}

// Search returns the count of fraud labels among the IVFSQ_TopK nearest
// reference vectors. Per-query work: K float32 centroid distances, query
// quantization, then a scan of NProbe clusters using int distance.
func (ivf *IVFSQ) Search(query [IVFSQ_Dim]float32) int {
	// 1. Find the NProbe nearest IVF centroids (float32 distance).
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

	// 2. Quantize the query once.
	var qInt [IVFSQ_Dim]int32
	for j := 0; j < IVFSQ_Dim; j++ {
		qInt[j] = int32(quantize(query[j]))
	}

	// 3. Scan candidate clusters with int64 distance accumulator.
	type cand struct {
		dist  int64
		fraud bool
	}
	var top [IVFSQ_TopK]cand
	for i := range top {
		top[i].dist = math.MaxInt64
	}

	// threshold caches the current 5th-best distance so the hot inner loop
	// can bail out as soon as a partial squared sum exceeds it. Most
	// candidates fall far from the query; per-dim early exit lets us skip
	// the remaining dims (and the int64 multiplies) for those candidates.
	threshold := top[IVFSQ_TopK-1].dist

	for _, probe := range topProbes {
		c := probe.idx
		start := ivf.ClusterOffsets[c]
		end := ivf.ClusterOffsets[c+1]
		for i := start; i < end; i++ {
			base := int(i) * IVFSQ_Dim

			d := qInt[0] - int32(ivf.Quantized[base+0])
			sum := int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[1] - int32(ivf.Quantized[base+1])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[2] - int32(ivf.Quantized[base+2])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[3] - int32(ivf.Quantized[base+3])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[4] - int32(ivf.Quantized[base+4])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[5] - int32(ivf.Quantized[base+5])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[6] - int32(ivf.Quantized[base+6])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[7] - int32(ivf.Quantized[base+7])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[8] - int32(ivf.Quantized[base+8])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[9] - int32(ivf.Quantized[base+9])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[10] - int32(ivf.Quantized[base+10])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[11] - int32(ivf.Quantized[base+11])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[12] - int32(ivf.Quantized[base+12])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}
			d = qInt[13] - int32(ivf.Quantized[base+13])
			sum += int64(d) * int64(d)
			if sum >= threshold {
				continue
			}

			fraud := ivf.Labels[i] == 1
			k := IVFSQ_TopK - 1
			for k > 0 && top[k-1].dist > sum {
				top[k] = top[k-1]
				k--
			}
			top[k] = cand{dist: sum, fraud: fraud}
			threshold = top[IVFSQ_TopK-1].dist
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
