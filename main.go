package main

import (
	"flag"
	"log"
	"net"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"github.com/lealre/rinha26-go/internal/fdpass"
	"github.com/lealre/rinha26-go/internal/ivfsq"
	"github.com/lealre/rinha26-go/internal/rawhttp"
)

func main() {
	resourcesDir := flag.String("resources-dir", "./resources", "directory containing normalization.json and mcc_risk.json")
	indexPath := flag.String("index", "./index.bin", "path to the prebuilt IVFSQ index file")
	listenAddr := flag.String("listen", "", "HTTP listen address: 'host:port' for TCP, or a path starting with '/' for a Unix domain socket. Empty means no direct HTTP listener.")
	listenFd := flag.String("listen-fd", "", "Unix control socket path for SCM_RIGHTS fd-passing from cmd/lb. Empty means no fd-passing listener.")
	flag.Parse()

	if *listenAddr == "" && *listenFd == "" {
		log.Fatalf("must set at least one of --listen or --listen-fd")
	}

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
	idx, err := ivfsq.LoadIndex(*indexPath)
	if err != nil {
		log.Fatalf("LoadIndex: %v", err)
	}
	log.Printf("index loaded: N=%d, K=%d in %s",
		idx.N, idx.K, time.Since(t0))

	ready := &atomic.Bool{}
	ready.Store(true)

	app := &App{
		Config: cfg,
		MCC:    mcc,
		Index:  idx,
		Ready:  ready,
	}

	// pprof stays on net/http (separate from the main /fraud-score path).
	// It's localhost-only, used to generate PGO profiles at build time.
	go func() {
		log.Printf("pprof listening on :6060")
		if err := http.ListenAndServe("localhost:6060", nil); err != nil {
			log.Printf("pprof server: %v", err)
		}
	}()

	srv := &rawhttp.Server{Handler: app}

	// Optional direct HTTP listener (UDS/TCP). Used pre-Option-D when an
	// external LB connects directly. Skip when only fd-passing is enabled.
	if *listenAddr != "" {
		listener, err := openListener(*listenAddr)
		if err != nil {
			log.Fatalf("listen %s: %v", *listenAddr, err)
		}
		defer listener.Close()
		log.Printf("listening on %s", listener.Addr())
		go func() {
			log.Fatal(srv.Serve(listener))
		}()
	}

	// Optional fd-passing listener (Option D). Foreground; blocks here.
	if *listenFd != "" {
		log.Printf("fd-passing receiver at %s", *listenFd)
		log.Fatal(fdpass.Serve(*listenFd, srv))
	}

	// If only --listen was set, sleep forever.
	select {}
}

// openListener returns a TCP or Unix-domain-socket listener based on addr.
// A leading '/' means Unix domain socket (path); anything else is treated as
// a TCP listen address (e.g. ":9999" or "localhost:9999").
//
// For UDS, the socket file is chmod'd 0666 so HAProxy or any other client
// running under a different uid can connect. A stale socket file from a
// prior run is removed first.
func openListener(addr string) (net.Listener, error) {
	if strings.HasPrefix(addr, "/") {
		_ = os.Remove(addr)
		l, err := net.Listen("unix", addr)
		if err != nil {
			return nil, err
		}
		if err := os.Chmod(addr, 0666); err != nil {
			_ = l.Close()
			return nil, err
		}
		return l, nil
	}
	return net.Listen("tcp", addr)
}
