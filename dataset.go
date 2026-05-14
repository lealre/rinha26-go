package main

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"
)

// Dataset holds the immutable in-memory reference set.
// After loadDataset returns, this struct is never mutated, so concurrent
// readers (HTTP handlers) need no synchronization.
type Dataset struct {
	Vectors [][14]float32
	Labels  []bool // true = fraud, false = legit
}

func loadDataset(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open references.json.gz: %w", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("gzip reader: %w", err)
	}
	defer gz.Close()

	dec := json.NewDecoder(gz)

	// Expect opening '['
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("read opening token: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return nil, fmt.Errorf("expected '[' at start, got %v", tok)
	}

	const expected = 3_000_000
	vectors := make([][14]float32, 0, expected)
	labels := make([]bool, 0, expected)

	type entry struct {
		Vector [14]float32 `json:"vector"`
		Label  string      `json:"label"`
	}

	for dec.More() {
		var e entry
		if err := dec.Decode(&e); err != nil {
			return nil, fmt.Errorf("decode reference entry %d: %w", len(vectors), err)
		}
		vectors = append(vectors, e.Vector)
		labels = append(labels, e.Label == "fraud")
	}

	// Expect closing ']'
	tok, err = dec.Token()
	if err != nil {
		return nil, fmt.Errorf("read closing token: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != ']' {
		return nil, fmt.Errorf("expected ']' at end, got %v", tok)
	}

	return &Dataset{Vectors: vectors, Labels: labels}, nil
}
