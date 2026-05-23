// build-index is a one-shot CLI that reads references.json.gz, trains the
// IVFPQ index, and writes /index.bin in the binary format defined in the spec.
//
// Run during `docker build`. NOT used at runtime.
package main

import (
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/lealre/rinha26-go/internal/ivfpq"
)

func main() {
	in := flag.String("in", "resources/references.json.gz", "path to references.json.gz")
	out := flag.String("out", "index.bin", "path to write the binary index")
	seed := flag.Int64("seed", 42, "RNG seed for k-means")
	flag.Parse()

	log.Printf("build-index: reading %s", *in)
	t0 := time.Now()
	vectors, labels, err := streamDataset(*in)
	if err != nil {
		log.Fatalf("streamDataset: %v", err)
	}
	log.Printf("build-index: loaded %d vectors in %s", len(labels), time.Since(t0))

	log.Printf("build-index: training IVFPQ index (seed=%d)...", *seed)
	tBuild := time.Now()
	idx := ivfpq.Build(vectors, labels, *seed)
	log.Printf("build-index: trained in %s", time.Since(tBuild))

	log.Printf("build-index: writing %s", *out)
	if err := idx.Save(*out); err != nil {
		log.Fatalf("Save: %v", err)
	}
	info, err := os.Stat(*out)
	if err != nil {
		log.Fatalf("stat %s: %v", *out, err)
	}
	log.Printf("build-index: wrote %s (%d bytes)", *out, info.Size())
}

// streamDataset reads references.json.gz token-by-token and returns a flat
// float32 slice of length N*Dim plus a parallel uint8 labels slice.
func streamDataset(path string) ([]float32, []uint8, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, fmt.Errorf("gzip reader: %w", err)
	}
	defer gz.Close()

	dec := json.NewDecoder(gz)

	tok, err := dec.Token()
	if err != nil {
		return nil, nil, fmt.Errorf("read opening token: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return nil, nil, fmt.Errorf("expected '[' at start, got %v", tok)
	}

	const expected = 3_000_000
	vectors := make([]float32, 0, expected*ivfpq.IVFPQ_Dim)
	labels := make([]uint8, 0, expected)

	type entry struct {
		Vector [ivfpq.IVFPQ_Dim]float32 `json:"vector"`
		Label  string                   `json:"label"`
	}
	for dec.More() {
		var e entry
		if err := dec.Decode(&e); err != nil {
			return nil, nil, fmt.Errorf("decode entry %d: %w", len(labels), err)
		}
		vectors = append(vectors, e.Vector[:]...)
		if e.Label == "fraud" {
			labels = append(labels, 1)
		} else {
			labels = append(labels, 0)
		}
	}

	tok, err = dec.Token()
	if err != nil {
		return nil, nil, fmt.Errorf("read closing token: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != ']' {
		return nil, nil, fmt.Errorf("expected ']' at end, got %v", tok)
	}
	return vectors, labels, nil
}
