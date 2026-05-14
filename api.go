package main

import (
	"encoding/json"
	"net/http"
	"sync/atomic"
)

// App owns the immutable shared state of the API.
// All fields except Ready are set once at startup and never mutated.
type App struct {
	Config  *Config
	MCC     map[string]float64
	Dataset *Dataset
	Ready   *atomic.Bool
}

type fraudResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float32 `json:"fraud_score"`
}

func (a *App) handleReady(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if a.Ready.Load() {
		w.WriteHeader(http.StatusOK)
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
}

func (a *App) handleFraudScore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var p Payload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}

	query, ok := vectorize(&p, a.Config, a.MCC)
	if !ok {
		http.Error(w, "invalid timestamp", http.StatusBadRequest)
		return
	}

	frauds := topKFraudCount(query, a.Dataset.Vectors, a.Dataset.Labels)
	score := float32(frauds) / 5.0

	resp := fraudResponse{
		Approved:   score < 0.6,
		FraudScore: score,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
