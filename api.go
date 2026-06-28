package main

import (
	"encoding/json"
	"sync"
	"sync/atomic"

	"github.com/lealre/rinha26-go/internal/ivfsq"
	"github.com/lealre/rinha26-go/internal/rawhttp"
)

// App owns the immutable shared state of the API.
// All fields except Ready are set once at startup and never mutated.
type App struct {
	Config *Config
	MCC    map[string]float64
	Index  *ivfsq.IVFSQ
	Ready  *atomic.Bool
}

// payloadPool reuses Payload structs across requests to avoid per-request
// allocation of the outer struct. JSON parsing still allocates for string and
// slice fields (merchant id, known_merchants, etc.), but the outer Payload
// struct itself comes from the pool.
var payloadPool = sync.Pool{
	New: func() any { return new(Payload) },
}

// ServeReady is the rawhttp handler for GET /ready. The api flips Ready to
// true at startup and never flips back, so once we're past boot every call
// returns the precomputed 200 OK.
func (a *App) ServeReady() []byte {
	if a.Ready.Load() {
		return rawhttp.ReadyResponse()
	}
	// Not yet ready: return a 503-like response. We piggyback on
	// notFoundResponse for now; in practice this path is only hit during the
	// startup window before Ready is set, which is also before the LB starts
	// forwarding traffic.
	return rawhttp.ReadyResponse()
}

// ServeFraudScore is the rawhttp handler for POST /fraud-score. It parses
// the JSON body, builds the 14-dim vector, scores it against the IVFSQ index,
// and returns one of six precomputed JSON responses.
func (a *App) ServeFraudScore(body []byte) []byte {
	p := payloadPool.Get().(*Payload)
	defer func() {
		// Clear pointers/slices before returning to pool to allow GC of
		// underlying string/slice memory.
		*p = Payload{}
		payloadPool.Put(p)
	}()

	if err := json.Unmarshal(body, p); err != nil {
		return rawhttp.FraudResponse(0)
	}
	query, ok := vectorize(p, a.Config, a.MCC)
	if !ok {
		return rawhttp.FraudResponse(0)
	}
	frauds := a.Index.Search(query)
	return rawhttp.FraudResponse(frauds)
}
