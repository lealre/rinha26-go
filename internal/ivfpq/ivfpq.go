package ivfpq

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"unsafe"
)

// IVFPQ parameter constants. Changing any of these requires bumping the
// IVFPQ_Magic version string so old index files refuse to load.
const (
	IVFPQ_K     = 1024 // number of IVF clusters
	IVFPQ_M     = 7    // number of PQ sub-vectors (must divide IVFPQ_Dim)
	IVFPQ_Kstar = 256  // PQ codebook size per sub-vector (fits in 1 byte)
	IVFPQ_Dim   = 14   // full vector dimensionality (the challenge's 14 dims)
	IVFPQ_SubD  = 2    // dimensions per PQ sub-vector (IVFPQ_Dim / IVFPQ_M)

	IVFPQ_TopK   = 5 // number of nearest neighbors to retrieve
	IVFPQ_NProbe = 8 // number of IVF clusters to probe per query
)

// IVFPQ_Magic is the 8-byte file-format identifier written to /index.bin.
// Bump the version suffix ("01") on any layout change.
var IVFPQ_Magic = [8]byte{'I', 'V', 'F', 'P', 'Q', 'v', '0', '1'}

// IVFPQ is the in-memory IVFPQ index.
//
// All four bulk slices are read directly from /index.bin via unsafe-slice
// reinterpretation; they share the underlying byte buffer with the file
// data the loader retains.
type IVFPQ struct {
	K     uint32 // = IVFPQ_K
	M     uint32 // = IVFPQ_M
	Kstar uint32 // = IVFPQ_Kstar
	N     uint32 // total number of indexed vectors

	IVFCentroids   []float32 // K * IVFPQ_Dim
	PQCodebooks    []float32 // M * Kstar * IVFPQ_SubD
	ClusterOffsets []uint32  // K+1; cluster c lives at Codes[Offsets[c]*M : Offsets[c+1]*M]
	Codes          []uint8   // N * M, grouped by cluster
	Labels         []uint8   // N (0 = legit, 1 = fraud)
}

// Save serializes the IVFPQ index to path using the binary layout defined
// in docs/superpowers/specs/2026-05-23-ann-ivf-pq-design.md.
func (ivf *IVFPQ) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)

	if _, err := w.Write(IVFPQ_Magic[:]); err != nil {
		return fmt.Errorf("write magic: %w", err)
	}
	hdr := []uint32{ivf.K, ivf.M, ivf.Kstar, ivf.N}
	if err := binary.Write(w, binary.LittleEndian, hdr); err != nil {
		return fmt.Errorf("write header: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, ivf.IVFCentroids); err != nil {
		return fmt.Errorf("write IVFCentroids: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, ivf.PQCodebooks); err != nil {
		return fmt.Errorf("write PQCodebooks: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, ivf.ClusterOffsets); err != nil {
		return fmt.Errorf("write ClusterOffsets: %w", err)
	}
	if _, err := w.Write(ivf.Codes); err != nil {
		return fmt.Errorf("write Codes: %w", err)
	}
	if _, err := w.Write(ivf.Labels); err != nil {
		return fmt.Errorf("write Labels: %w", err)
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush: %w", err)
	}
	return nil
}

// LoadIndex reads an IVFPQ index from path. The returned *IVFPQ holds slices
// that alias the underlying byte buffer returned by os.ReadFile; the buffer
// stays alive as long as any of those slices is reachable.
//
// Assumes little-endian host (amd64). The challenge spec mandates amd64.
func LoadIndex(path string) (*IVFPQ, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if len(data) < 24 {
		return nil, fmt.Errorf("file %s too small: %d bytes", path, len(data))
	}

	var magic [8]byte
	copy(magic[:], data[0:8])
	if magic != IVFPQ_Magic {
		return nil, fmt.Errorf("bad magic in %s: got %q, want %q",
			path, magic[:], IVFPQ_Magic[:])
	}

	ivf := &IVFPQ{
		K:     binary.LittleEndian.Uint32(data[8:12]),
		M:     binary.LittleEndian.Uint32(data[12:16]),
		Kstar: binary.LittleEndian.Uint32(data[16:20]),
		N:     binary.LittleEndian.Uint32(data[20:24]),
	}
	if ivf.K != IVFPQ_K || ivf.M != IVFPQ_M || ivf.Kstar != IVFPQ_Kstar {
		return nil, fmt.Errorf("unsupported params in %s: K=%d M=%d Kstar=%d (want %d/%d/%d)",
			path, ivf.K, ivf.M, ivf.Kstar, IVFPQ_K, IVFPQ_M, IVFPQ_Kstar)
	}

	offset := 24

	centroidsBytes := int(ivf.K) * IVFPQ_Dim * 4
	if offset+centroidsBytes > len(data) {
		return nil, fmt.Errorf("file truncated at IVFCentroids")
	}
	ivf.IVFCentroids = bytesToFloat32(data[offset : offset+centroidsBytes])
	offset += centroidsBytes

	codebooksBytes := int(ivf.M) * int(ivf.Kstar) * IVFPQ_SubD * 4
	if offset+codebooksBytes > len(data) {
		return nil, fmt.Errorf("file truncated at PQCodebooks")
	}
	ivf.PQCodebooks = bytesToFloat32(data[offset : offset+codebooksBytes])
	offset += codebooksBytes

	offsetsBytes := (int(ivf.K) + 1) * 4
	if offset+offsetsBytes > len(data) {
		return nil, fmt.Errorf("file truncated at ClusterOffsets")
	}
	ivf.ClusterOffsets = bytesToUint32(data[offset : offset+offsetsBytes])
	offset += offsetsBytes

	codesBytes := int(ivf.N) * int(ivf.M)
	if offset+codesBytes > len(data) {
		return nil, fmt.Errorf("file truncated at Codes")
	}
	ivf.Codes = data[offset : offset+codesBytes]
	offset += codesBytes

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

// bytesToFloat32 reinterprets a byte slice as a float32 slice without copying.
// Caller must ensure the input length is a multiple of 4 and 4-byte aligned
// (our /index.bin layout is designed so every section starts at a 4-byte offset).
func bytesToFloat32(b []byte) []float32 {
	if len(b)%4 != 0 {
		panic("bytesToFloat32: length not multiple of 4")
	}
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// bytesToUint32 reinterprets a byte slice as a uint32 slice without copying.
func bytesToUint32(b []byte) []uint32 {
	if len(b)%4 != 0 {
		panic("bytesToUint32: length not multiple of 4")
	}
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// Build trains the IVFPQ index on the given dataset and returns the populated
// struct. Called only by cmd/build-index (offline), never at runtime.
//
// vectors: flat row-major float32 slice of length N*Dim.
// labels:  flat byte slice of length N (0 = legit, 1 = fraud).
// seed:    RNG seed; same seed → same trained index across rebuilds.
func Build(vectors []float32, labels []uint8, seed int64) *IVFPQ {
	n := len(labels)
	if len(vectors) != n*IVFPQ_Dim {
		log.Fatalf("Build: vectors length %d does not match N*Dim = %d*%d = %d",
			len(vectors), n, IVFPQ_Dim, n*IVFPQ_Dim)
	}

	log.Printf("Build: %d vectors", n)

	// Step 1: Train IVF centroids on the full dataset.
	log.Printf("Build: training IVF centroids (K=%d, dim=%d)...", IVFPQ_K, IVFPQ_Dim)
	ivfRes := KMeans(vectors, n, IVFPQ_Dim, IVFPQ_K, 30, seed)
	log.Printf("Build: IVF centroids done")

	// Step 2: Train PQ codebooks, one per sub-position.
	codebooks := make([]float32, IVFPQ_M*IVFPQ_Kstar*IVFPQ_SubD)
	for m := 0; m < IVFPQ_M; m++ {
		// Extract sub-position m's 2-D slice from every vector.
		sub := make([]float32, n*IVFPQ_SubD)
		for i := 0; i < n; i++ {
			for d := 0; d < IVFPQ_SubD; d++ {
				sub[i*IVFPQ_SubD+d] = vectors[i*IVFPQ_Dim+m*IVFPQ_SubD+d]
			}
		}
		log.Printf("Build: training PQ codebook %d/%d", m+1, IVFPQ_M)
		pqRes := KMeans(sub, n, IVFPQ_SubD, IVFPQ_Kstar, 30, seed+int64(m+1))
		copy(codebooks[m*IVFPQ_Kstar*IVFPQ_SubD:(m+1)*IVFPQ_Kstar*IVFPQ_SubD], pqRes.Centroids)
	}
	log.Printf("Build: PQ codebooks done")

	// Step 3: Count cluster sizes from the IVF assignments.
	counts := make([]uint32, IVFPQ_K)
	for _, c := range ivfRes.Assignments {
		counts[c]++
	}
	offsets := make([]uint32, IVFPQ_K+1)
	for c := 0; c < IVFPQ_K; c++ {
		offsets[c+1] = offsets[c] + counts[c]
	}
	if offsets[IVFPQ_K] != uint32(n) {
		log.Fatalf("Build: offset sum mismatch: got %d, want %d", offsets[IVFPQ_K], n)
	}

	// Step 4: Fill Codes and Labels in cluster-grouped order.
	codes := make([]uint8, n*IVFPQ_M)
	labs := make([]uint8, n)
	cursor := make([]uint32, IVFPQ_K)
	copy(cursor, offsets[:IVFPQ_K])

	log.Printf("Build: encoding %d vectors into PQ codes...", n)
	for i := 0; i < n; i++ {
		c := ivfRes.Assignments[i]
		pos := cursor[c]
		cursor[c]++

		// Encode each sub-vector.
		for m := 0; m < IVFPQ_M; m++ {
			start := i*IVFPQ_Dim + m*IVFPQ_SubD
			subVec := vectors[start : start+IVFPQ_SubD]
			cbStart := m * IVFPQ_Kstar * IVFPQ_SubD
			cb := codebooks[cbStart : cbStart+IVFPQ_Kstar*IVFPQ_SubD]
			codes[int(pos)*IVFPQ_M+m] = uint8(nearest(subVec, cb, IVFPQ_Kstar, IVFPQ_SubD))
		}
		labs[pos] = labels[i]
	}
	log.Printf("Build: done")

	return &IVFPQ{
		K:              IVFPQ_K,
		M:              IVFPQ_M,
		Kstar:          IVFPQ_Kstar,
		N:              uint32(n),
		IVFCentroids:   ivfRes.Centroids,
		PQCodebooks:    codebooks,
		ClusterOffsets: offsets,
		Codes:          codes,
		Labels:         labs,
	}
}

// Search runs an IVFPQ query and returns the count of fraud labels (1) among
// the IVFPQ_TopK nearest reference vectors (by approximate PQ distance).
//
// Per-query work: distance to K IVF centroids + LUT build + scan of nprobe
// clusters using LUT-based distance.
func (ivf *IVFPQ) Search(query [IVFPQ_Dim]float32) int {
	// 1. Find the nprobe nearest IVF centroids.
	type centDist struct {
		idx  uint32
		dist float32
	}
	var topProbes [IVFPQ_NProbe]centDist
	for i := range topProbes {
		topProbes[i].dist = float32(1e30)
	}
	for c := uint32(0); c < ivf.K; c++ {
		var d float32
		base := int(c) * IVFPQ_Dim
		for j := 0; j < IVFPQ_Dim; j++ {
			diff := query[j] - ivf.IVFCentroids[base+j]
			d += diff * diff
		}
		if d >= topProbes[IVFPQ_NProbe-1].dist {
			continue
		}
		k := IVFPQ_NProbe - 1
		for k > 0 && topProbes[k-1].dist > d {
			topProbes[k] = topProbes[k-1]
			k--
		}
		topProbes[k] = centDist{idx: c, dist: d}
	}

	// 2. Build the M LUTs.
	var lut [IVFPQ_M][IVFPQ_Kstar]float32
	for m := 0; m < IVFPQ_M; m++ {
		qs0 := query[m*IVFPQ_SubD+0]
		qs1 := query[m*IVFPQ_SubD+1]
		base := m * IVFPQ_Kstar * IVFPQ_SubD
		for k := 0; k < IVFPQ_Kstar; k++ {
			d0 := qs0 - ivf.PQCodebooks[base+k*IVFPQ_SubD+0]
			d1 := qs1 - ivf.PQCodebooks[base+k*IVFPQ_SubD+1]
			lut[m][k] = d0*d0 + d1*d1
		}
	}

	// 3. Scan the nprobe clusters, maintaining a top-K=5 of candidates.
	type cand struct {
		dist  float32
		fraud bool
	}
	var top [IVFPQ_TopK]cand
	for i := range top {
		top[i].dist = float32(1e30)
	}

	for _, probe := range topProbes {
		c := probe.idx
		start := ivf.ClusterOffsets[c]
		end := ivf.ClusterOffsets[c+1]
		for i := start; i < end; i++ {
			codeBase := int(i) * IVFPQ_M
			d := lut[0][ivf.Codes[codeBase+0]]
			d += lut[1][ivf.Codes[codeBase+1]]
			d += lut[2][ivf.Codes[codeBase+2]]
			d += lut[3][ivf.Codes[codeBase+3]]
			d += lut[4][ivf.Codes[codeBase+4]]
			d += lut[5][ivf.Codes[codeBase+5]]
			d += lut[6][ivf.Codes[codeBase+6]]
			if d >= top[IVFPQ_TopK-1].dist {
				continue
			}
			fraud := ivf.Labels[i] == 1
			k := IVFPQ_TopK - 1
			for k > 0 && top[k-1].dist > d {
				top[k] = top[k-1]
				k--
			}
			top[k] = cand{dist: d, fraud: fraud}
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
