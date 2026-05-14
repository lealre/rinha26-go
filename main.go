package main

import (
	"flag"
	"log"
	"net/http"
	"path/filepath"
	"sync/atomic"
	"time"
)

func main() {
	resourcesDir := flag.String("resources-dir", "./resources", "directory containing references.json.gz, mcc_risk.json, normalization.json")
	flag.Parse()

	cfgPath := filepath.Join(*resourcesDir, "normalization.json")
	mccPath := filepath.Join(*resourcesDir, "mcc_risk.json")
	refsPath := filepath.Join(*resourcesDir, "references.json.gz")

	log.Printf("loading config from %s", cfgPath)
	cfg, err := loadConfig(cfgPath)
	if err != nil {
		log.Fatalf("loadConfig: %v", err)
	}

	log.Printf("loading mcc risk from %s", mccPath)
	mcc, err := loadMCCRisk(mccPath)
	if err != nil {
		log.Fatalf("loadMCCRisk: %v", err)
	}

	log.Printf("loading dataset from %s (this takes ~10-30s)", refsPath)
	t0 := time.Now()
	ds, err := loadDataset(refsPath)
	if err != nil {
		log.Fatalf("loadDataset: %v", err)
	}
	log.Printf("dataset loaded: %d vectors in %s", len(ds.Vectors), time.Since(t0))

	ready := &atomic.Bool{}
	ready.Store(true)

	app := &App{
		Config:  cfg,
		MCC:     mcc,
		Dataset: ds,
		Ready:   ready,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ready", app.handleReady)
	mux.HandleFunc("/fraud-score", app.handleFraudScore)

	addr := ":9999"
	log.Printf("listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}
