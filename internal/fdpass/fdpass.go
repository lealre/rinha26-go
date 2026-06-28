// Package fdpass implements the api-side receiver for SCM_RIGHTS
// file-descriptor passing from cmd/lb.
//
// Each api worker binds a Unix control socket. The LB connects to it and
// sends raw client-socket fds via SCM_RIGHTS; this package reads them off
// the control socket, wraps each fd as a net.Conn, and hands it to a
// caller-provided ConnHandler (typically *rawhttp.Server.ServeConn).
package fdpass

import (
	"log"
	"net"
	"os"
	"syscall"
)

// ConnHandler is the api-side endpoint that does the actual HTTP work for
// a single received connection.
type ConnHandler interface {
	ServeConn(net.Conn)
}

// Serve binds the control socket at path, accepts LB connections, and
// dispatches received fds to handler. Blocks until the listener errors.
// chmod 0666 on the socket so the LB can connect as a different uid.
func Serve(path string, handler ConnHandler) error {
	_ = os.Remove(path)
	ln, err := net.Listen("unix", path)
	if err != nil {
		return err
	}
	if err := os.Chmod(path, 0666); err != nil {
		_ = ln.Close()
		return err
	}
	log.Printf("fdpass: listening on %s", path)

	for {
		c, err := ln.Accept()
		if err != nil {
			return err
		}
		uc, ok := c.(*net.UnixConn)
		if !ok {
			_ = c.Close()
			continue
		}
		go drain(uc, handler)
	}
}

// drain reads SCM_RIGHTS messages off ctrl in a loop, parsing each into
// one or more passed fds and starting a handler goroutine per fd.
func drain(ctrl *net.UnixConn, handler ConnHandler) {
	defer ctrl.Close()
	buf := make([]byte, 16)
	// CmsgSpace(4*8) gives room for up to 8 fds per message — far more
	// than we ever send (current LB sends 1 fd per message).
	oob := make([]byte, syscall.CmsgSpace(4*8))
	for {
		_, oobn, _, _, err := ctrl.ReadMsgUnix(buf, oob)
		if err != nil {
			return
		}
		if oobn == 0 {
			continue
		}
		msgs, err := syscall.ParseSocketControlMessage(oob[:oobn])
		if err != nil {
			continue
		}
		for i := range msgs {
			fds, err := syscall.ParseUnixRights(&msgs[i])
			if err != nil {
				continue
			}
			for _, fd := range fds {
				go serveOne(fd, handler)
			}
		}
	}
}

// serveOne wraps a received fd as a net.Conn and runs the handler.
func serveOne(fd int, handler ConnHandler) {
	f := os.NewFile(uintptr(fd), "passed-fd")
	if f == nil {
		_ = syscall.Close(fd)
		return
	}
	conn, err := net.FileConn(f)
	// FileConn dups the fd; close our handle to release the SCM_RIGHTS copy.
	_ = f.Close()
	if err != nil {
		return
	}
	handler.ServeConn(conn)
}
