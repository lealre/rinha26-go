// Package rawhttp is a minimal HTTP/1.1 server tuned for this api's hot path.
// It hand-parses requests from a pooled byte buffer and dispatches to two
// precomputed-response endpoints (POST /fraud-score, GET /ready). The
// steady-state per-request hot path allocates zero heap objects.
package rawhttp

import (
	"bytes"
	"io"
	"net"
	"strconv"
	"sync"
	"time"
)

// Handler is the application-side dispatcher. ServeFraudScore receives the
// raw POST body (JSON) and returns the full HTTP response bytes to write back
// (status line + headers + body). ServeReady is parameterless and returns the
// precomputed /ready response.
type Handler interface {
	ServeFraudScore(body []byte) []byte
	ServeReady() []byte
}

const (
	readBufSize    = 4096
	maxRequestSize = 8 * 1024
	readTimeout    = 30 * time.Second
)

var (
	fraudResponses   [6][]byte
	readyResponse    []byte
	notFoundResponse []byte
	methodNotAllowed []byte
	badRequest       []byte
	headerEndMarker  = []byte("\r\n\r\n")
)

func init() {
	// Precompute the 6 possible /fraud-score response bodies. The api can
	// only return fraud_score ∈ {0/5, 1/5, 2/5, 3/5, 4/5, 5/5} = {0.0, 0.2,
	// 0.4, 0.6, 0.8, 1.0} and approved is fraud_score < 0.6 (see api.go).
	bodies := [6]string{
		`{"approved":true,"fraud_score":0}`,
		`{"approved":true,"fraud_score":0.2}`,
		`{"approved":true,"fraud_score":0.4}`,
		`{"approved":false,"fraud_score":0.6}`,
		`{"approved":false,"fraud_score":0.8}`,
		`{"approved":false,"fraud_score":1}`,
	}
	for i, body := range bodies {
		fraudResponses[i] = buildResponse(200, "OK", "application/json", body)
	}
	readyResponse = buildResponse(200, "OK", "", "")
	notFoundResponse = buildResponse(404, "Not Found", "", "")
	methodNotAllowed = buildResponse(405, "Method Not Allowed", "", "")
	badRequest = buildResponse(400, "Bad Request", "", "")
}

// buildResponse returns a full HTTP/1.1 response byte slice ready to be
// written to a connection.
func buildResponse(code int, text, ct, body string) []byte {
	hdr := "HTTP/1.1 " + strconv.Itoa(code) + " " + text + "\r\nConnection: keep-alive\r\n"
	if ct != "" {
		hdr += "Content-Type: " + ct + "\r\n"
	}
	hdr += "Content-Length: " + strconv.Itoa(len(body)) + "\r\n\r\n"
	out := make([]byte, len(hdr)+len(body))
	copy(out, hdr)
	copy(out[len(hdr):], body)
	return out
}

// FraudResponse returns the precomputed HTTP response for fraud_count ∈ [0, 5].
func FraudResponse(count int) []byte {
	if count < 0 || count >= len(fraudResponses) {
		return fraudResponses[0]
	}
	return fraudResponses[count]
}

// ReadyResponse returns the precomputed /ready response.
func ReadyResponse() []byte {
	return readyResponse
}

// bufPool keeps per-connection 4 KB read buffers; the typical request fits in
// one read so the pool re-use ratio is high.
var bufPool = sync.Pool{
	New: func() any {
		b := make([]byte, readBufSize)
		return &b
	},
}

// Server is a minimal HTTP/1.1 server.
type Server struct {
	Handler Handler
}

// Serve runs the accept loop, spawning one goroutine per accepted connection.
// Returns when the listener is closed.
func (s *Server) Serve(ln net.Listener) error {
	for {
		conn, err := ln.Accept()
		if err != nil {
			return err
		}
		go s.ServeConn(conn)
	}
}

// ServeConn drains a single connection: parse request → dispatch → write
// response, then loop for the next pipelined / keep-alive request.
func (s *Server) ServeConn(conn net.Conn) {
	defer conn.Close()

	if tcp, ok := conn.(*net.TCPConn); ok {
		_ = tcp.SetNoDelay(true)
	}

	bp := bufPool.Get().(*[]byte)
	buf := *bp
	defer bufPool.Put(bp)

	pos, used := 0, 0

	for {
		// Read until "\r\n\r\n" is in buf[pos:used].
		headEnd := bytes.Index(buf[pos:used], headerEndMarker)
		for headEnd < 0 {
			if used == len(buf) {
				if pos > 0 {
					copy(buf, buf[pos:used])
					used -= pos
					pos = 0
				} else {
					return
				}
			}
			_ = conn.SetReadDeadline(time.Now().Add(readTimeout))
			n, err := conn.Read(buf[used:])
			if n > 0 {
				used += n
				if used-pos > maxRequestSize {
					_, _ = conn.Write(badRequest)
					return
				}
				headEnd = bytes.Index(buf[pos:used], headerEndMarker)
				continue
			}
			if err != nil {
				if err == io.EOF {
					return
				}
				return
			}
		}
		headEnd += pos + 4 // absolute index of byte after "\r\n\r\n"

		method, path, contentLen := parseRequestLine(buf[pos:headEnd])
		bodyEnd := headEnd + contentLen

		// Make sure we have the full body.
		for used < bodyEnd {
			if bodyEnd-pos > maxRequestSize {
				_, _ = conn.Write(badRequest)
				return
			}
			if bodyEnd > len(buf) {
				if pos > 0 {
					copy(buf, buf[pos:used])
					used -= pos
					bodyEnd -= pos
					headEnd -= pos
					pos = 0
				} else {
					_, _ = conn.Write(badRequest)
					return
				}
			}
			_ = conn.SetReadDeadline(time.Now().Add(readTimeout))
			n, err := conn.Read(buf[used:])
			if n > 0 {
				used += n
				continue
			}
			if err != nil {
				return
			}
		}

		// Dispatch.
		var resp []byte
		switch {
		case methodEquals(method, methodPOST) && pathEquals(path, pathFraudScore):
			resp = s.Handler.ServeFraudScore(buf[headEnd:bodyEnd])
		case methodEquals(method, methodGET) && pathEquals(path, pathReady):
			resp = s.Handler.ServeReady()
		default:
			resp = notFoundResponse
		}

		if _, err := conn.Write(resp); err != nil {
			return
		}

		// Advance past the request just served.
		pos = bodyEnd
		if pos >= used {
			pos, used = 0, 0
		}
	}
}

var (
	methodPOST     = []byte("POST")
	methodGET      = []byte("GET")
	pathFraudScore = []byte("/fraud-score")
	pathReady      = []byte("/ready")
)

func methodEquals(a, b []byte) bool { return bytes.Equal(a, b) }
func pathEquals(a, b []byte) bool   { return bytes.Equal(a, b) }

// parseRequestLine extracts METHOD and PATH from the request line as
// subslices of buf, and also scans buf for a Content-Length header.
// Defaults: nil method/path on malformed input; contentLen 0 when absent.
func parseRequestLine(buf []byte) (method, path []byte, contentLen int) {
	contentLen = findContentLength(buf)

	sp := bytes.IndexByte(buf, ' ')
	if sp <= 0 {
		return nil, nil, contentLen
	}
	method = buf[:sp]

	rest := buf[sp+1:]
	sp2 := bytes.IndexByte(rest, ' ')
	if sp2 <= 0 {
		return method, nil, contentLen
	}
	path = rest[:sp2]
	return method, path, contentLen
}

// findContentLength scans for a Content-Length header and parses its value.
// Returns 0 if not found.
func findContentLength(buf []byte) int {
	const hdr = "Content-Length:"
	// Walk line by line.
	for i := 0; i+len(hdr) < len(buf); i++ {
		if (buf[i] == '\n' || buf[i] == '\r') && i+1+len(hdr) <= len(buf) {
			// Check if next line starts with Content-Length (case-insensitive).
			start := i + 1
			if buf[start] == '\n' && start+1 < len(buf) {
				start++
			}
			if start+len(hdr) > len(buf) {
				continue
			}
			if !equalFold(buf[start:start+len(hdr)], []byte(hdr)) {
				continue
			}
			j := start + len(hdr)
			for j < len(buf) && (buf[j] == ' ' || buf[j] == '\t') {
				j++
			}
			n := 0
			for j < len(buf) && buf[j] >= '0' && buf[j] <= '9' {
				n = n*10 + int(buf[j]-'0')
				j++
			}
			return n
		}
	}
	// Also check the very first header (right after request line).
	for i := 0; i+len(hdr) < len(buf); i++ {
		if equalFold(buf[i:i+len(hdr)], []byte(hdr)) {
			j := i + len(hdr)
			for j < len(buf) && (buf[j] == ' ' || buf[j] == '\t') {
				j++
			}
			n := 0
			for j < len(buf) && buf[j] >= '0' && buf[j] <= '9' {
				n = n*10 + int(buf[j]-'0')
				j++
			}
			return n
		}
	}
	return 0
}

// equalFold is an ASCII case-insensitive equality test.
func equalFold(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		ca, cb := a[i], b[i]
		if ca >= 'A' && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if cb >= 'A' && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}
