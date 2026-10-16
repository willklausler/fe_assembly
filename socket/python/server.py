"""Material server for the socket paradigm: fe_jax behind a TCP socket.

One client, synchronous request/response, native byte order (same machine).
Request: int32 op, handle, n, nbytes, then nbytes of payload.
  op 1 make:     payload = model (16 chars) + cfg (4 chars) + par (doubles)
                 reply   = int64 handle (or -1)
  op 2 eval:     payload = F(3,3,n) in Fortran order
                 reply   = int64 status, then stress(ns,n) and tangent(nt,n) doubles
  op 3 noop:     like eval with ns, nt = 6, 36, but nothing is computed
  op 0 shutdown: no reply
The server exits on shutdown or when the client closes the connection.
Usage: python server.py port
"""
import os
import socket
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "jax", "python"))
import fe_jax  # noqa: E402

HEADER = struct.Struct("=4i")
SIZES = {"ref": (6, 36), "cur": (6, 36), "mix": (9, 81)}


def recv_exact(conn, buf):
    view = memoryview(buf)
    while view:
        got = conn.recv_into(view)
        if got == 0:
            raise ConnectionError("client closed the connection")
        view = view[got:]


def serve(port):
    with socket.create_server(("127.0.0.1", port)) as srv:
        conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    handles = []  # (run, par, ns, nt)
    header = bytearray(HEADER.size)
    with conn:
        while True:
            try:
                recv_exact(conn, header)
            except ConnectionError:
                return
            op, h, n, nbytes = HEADER.unpack(header)
            payload = bytearray(nbytes)
            recv_exact(conn, payload)
            if op == 0:
                return
            if op == 1:
                model = payload[:16].decode().strip()
                cfg = payload[16:20].decode().strip()
                try:
                    handles.append((fe_jax.make(model, cfg), payload[20:], *SIZES[cfg]))
                    conn.sendall(struct.pack("=q", len(handles) - 1))
                except Exception as err:  # report to the client instead of dying
                    print(f"server: make {model} {cfg}: {err!r}", file=sys.stderr)
                    conn.sendall(struct.pack("=q", -1))
                continue
            ns, nt = (6, 36) if op == 3 else handles[h][2:]
            # status, then stress and tangent, sent with one sendall
            reply = bytearray(8 * (1 + (ns + nt) * n))
            if op == 2:
                run, par = handles[h][:2]
                out = memoryview(reply)[8:]
                try:
                    run(memoryview(payload), memoryview(par), out[: 8 * ns * n], out[8 * ns * n :])
                except Exception as err:
                    print(f"server: eval handle {h}: {err!r}", file=sys.stderr)
                    reply[:8] = struct.pack("=q", -1)
            conn.sendall(reply)


if __name__ == "__main__":
    serve(int(sys.argv[1]))
