package main

import (
	"flag"
	"log"
	"net/http"
	_ "net/http/pprof"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/lealre/rinha26-go/internal/ivfpq"
)

func main() {
	resourcesDir := flag.String("resources-dir", "./resources", "directory containing normalization.json and mcc_risk.json")
	indexPath := flag.String("index", "./index.bin", "path to the prebuilt IVFPQ index file")
	flag.Parse()

	cfgPath := filepath.Join(*resourcesDir, "normalization.json")
	mccPath := filepath.Join(*resourcesDir, "mcc_risk.json")

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

	log.Printf("loading index from %s", *indexPath)
	t0 := time.Now()
	idx, err := ivfpq.LoadIndex(*indexPath)
	if err != nil {
		log.Fatalf("LoadIndex: %v", err)
	}
	log.Printf("index loaded: N=%d, K=%d, M=%d, Kstar=%d in %s",
		idx.N, idx.K, idx.M, idx.Kstar, time.Since(t0))

	ready := &atomic.Bool{}
	ready.Store(true)

	app := &App{
		Config: cfg,
		MCC:    mcc,
		Index:  idx,
		Ready:  ready,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ready", app.handleReady)
	mux.HandleFunc("/fraud-score", app.handleFraudScore)

	go func() {
		log.Printf("pprof listening on :6060")
		if err := http.ListenAndServe("localhost:6060", nil); err != nil {
			log.Printf("pprof server: %v", err)
		}
	}()

	addr := ":9999"
	log.Printf("listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}
