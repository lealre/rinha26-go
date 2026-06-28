// cmd/lb is a TCP-to-Unix-FD forwarding load balancer.
//
// It listens on a TCP port (default :9999), and for each accepted client
// connection hands the raw file descriptor to one of N configured worker
// control sockets via SCM_RIGHTS. After the handoff the kernel duplicates
// the fd into the worker's process table; the worker reads and writes the
// client socket directly, and this LB is out of the data path.
//
// Round-robin distribution. Workers must already be listening on their
// control sockets — the LB waits for them at startup (up to 60s).
package main

import (
	"flag"
	"log"
	"net"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

func main() {
	listenAddr := flag.String("listen", ":9999", "TCP listen address")
	workers := flag.String("workers",
		"/var/run/rinha/api1-ctrl.sock,/var/run/rinha/api2-ctrl.sock",
		"comma-separated worker control socket paths")
	flag.Parse()

	paths := strings.Split(*workers, ",")
	cons := make([]*net.UnixConn, 0, len(paths))
	for _, p := range paths {
		p = strings.TrimSpace(p)
		log.Printf("dialing worker control socket %s", p)
		c, err := dialUnixRetry(p, 60*time.Second)
		if err != nil {
			log.Fatalf("dial %s: %v", p, err)
		}
		cons = append(cons, c)
	}
	log.Printf("connected to %d workers", len(cons))

	ln, err := net.Listen("tcp", *listenAddr)
	if err != nil {
		log.Fatalf("listen %s: %v", *listenAddr, err)
	}

	// TCP_DEFER_ACCEPT: kernel only wakes accept() once data is ready.
	tcpLn := ln.(*net.TCPListener)
	if raw, err := tcpLn.SyscallConn(); err == nil {
		_ = raw.Control(func(fd uintptr) {
			_ = syscall.SetsockoptInt(int(fd),
				syscall.IPPROTO_TCP, syscall.TCP_DEFER_ACCEPT, 3)
		})
	}
	log.Printf("listening on %s", *listenAddr)

	var rr uint64
	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Printf("accept: %v", err)
			continue
		}
		tcp := conn.(*net.TCPConn)

		idx := atomic.AddUint64(&rr, 1) % uint64(len(cons))
		if err := handoff(tcp, cons[idx]); err != nil {
			log.Printf("handoff to worker %d: %v", idx, err)
		}
		// Close our copy of the fd; the kernel duplicated it into the
		// worker when SCM_RIGHTS was sent.
		_ = tcp.Close()
	}
}

// handoff sends tcp's underlying fd to ctrl via SCM_RIGHTS. After this
// call returns the worker has its own duplicate of the fd in its process
// table; we close ours in the caller.
func handoff(tcp *net.TCPConn, ctrl *net.UnixConn) error {
	raw, err := tcp.SyscallConn()
	if err != nil {
		return err
	}
	var sendErr error
	cerr := raw.Control(func(fd uintptr) {
		oob := syscall.UnixRights(int(fd))
		_, _, sendErr = ctrl.WriteMsgUnix([]byte{0}, oob, nil)
	})
	if cerr != nil {
		return cerr
	}
	return sendErr
}

// dialUnixRetry retries DialUnix until success or timeout. Workers take a
// few seconds to bind their control sockets at boot; the LB depends_on the
// workers being healthy but we still want resilience to brief flakes.
func dialUnixRetry(path string, timeout time.Duration) (*net.UnixConn, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		c, err := net.DialUnix("unix", nil,
			&net.UnixAddr{Name: path, Net: "unix"})
		if err == nil {
			return c, nil
		}
		lastErr = err
		time.Sleep(500 * time.Millisecond)
	}
	return nil, lastErr
}

