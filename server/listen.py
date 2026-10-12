"""How the server listens.

Without a certificate: one plain-HTTP socket, exactly as `uvicorn.run` does.

With SSL_CERTFILE / SSL_KEYFILE set and HOST a wildcard (0.0.0.0), the same
port serves two ways at once:

    0.0.0.0:PORT    HTTPS  for everything that arrives from the network
                           (the router's port forward, the LAN, the domain)
    127.0.0.1:PORT  HTTP   for this machine: the Chrome extension, `register`,
                           `start`'s health check, `tunnel`

Both sockets set SO_REUSEPORT, which lets them share the port; Linux hands a
connection to the socket bound to the most specific address, so anything
addressed to 127.0.0.1 gets plain HTTP and everything else gets TLS. The
local tools keep their http://127.0.0.1:PORT address, and the certificate,
which names the public domain, is never asked to cover "127.0.0.1".
LOCAL_HTTP=0 turns the loopback listener off (HTTPS only).

SO_REUSEPORT would also let a second copy of the server bind the port and
silently take half the connections, so the port is first probed with a plain
bind; a port already in use ends startup with an error, as before.

A renewed certificate is picked up without a restart: the files are checked
once a minute and reloaded into the live TLS context when they change.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import ssl
import threading
from typing import List, Optional, Tuple

import uvicorn

LOOPBACK = "127.0.0.1"
WILDCARDS = ("0.0.0.0", "")
CERT_CHECK_SECONDS = 60.0


class _Server(uvicorn.Server):
    """A uvicorn server that leaves signals to `serve`, which stops all of
    them together (uvicorn's own handler stops only the server that set it)."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


def local_http_enabled() -> bool:
    return os.getenv("LOCAL_HTTP", "1").strip().lower() not in ("0", "false", "no", "off")


def port_free(host: str, port: int) -> bool:
    """Whether `host:port` can be bound by a socket that does NOT share."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host or "0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def bind_shared(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host or "0.0.0.0", port))
    sock.set_inheritable(True)
    return sock


def check_cert_files(certfile: str, keyfile: str) -> None:
    """Fail early and clearly, instead of uvicorn's bare traceback."""
    for label, path in (("SSL_CERTFILE", certfile), ("SSL_KEYFILE", keyfile)):
        if not os.path.isfile(path):
            raise SystemExit(f"{label}={path} does not exist. Get a certificate with "
                             "./setup-https.sh, or remove SSL_CERTFILE/SSL_KEYFILE "
                             "from .env to serve plain HTTP.")
        if not os.access(path, os.R_OK):
            raise SystemExit(f"{label}={path} is not readable by this user.")
    try:
        ssl.create_default_context(ssl.Purpose.CLIENT_AUTH).load_cert_chain(certfile, keyfile)
    except (ssl.SSLError, ValueError) as exc:
        raise SystemExit(f"SSL_CERTFILE/SSL_KEYFILE do not form a valid pair: {exc}")


def build(app, host: str, port: int, certfile: str, keyfile: str,
          local_http: Optional[bool] = None,
          log_level: str = "info") -> List[Tuple[_Server, List[socket.socket], str]]:
    """The servers to run and the sockets each one serves, with a description."""
    check_cert_files(certfile, keyfile)
    if local_http is None:
        local_http = local_http_enabled()
    split = (local_http and host in WILDCARDS and hasattr(socket, "SO_REUSEPORT"))

    if not port_free(host, port):
        raise SystemExit(f"Port {port} is already in use; is another server running? "
                         "Stop it first (`start` does that for its own).")

    tls = _Server(uvicorn.Config(app, ssl_certfile=certfile, ssl_keyfile=keyfile,
                                 log_level=log_level))
    if not split:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host or "0.0.0.0", port))
        return [(tls, [sock], f"HTTPS on https://{host or '0.0.0.0'}:{port}")]

    # Bind the specific address first; the kernel prefers it either way.
    plain_sock = bind_shared(LOOPBACK, port)
    tls_sock = bind_shared(host, port)
    plain = _Server(uvicorn.Config(app, lifespan="off", log_level=log_level))
    return [
        (tls, [tls_sock], f"HTTPS on https://{host or '0.0.0.0'}:{port}"),
        (plain, [plain_sock], f"plain HTTP for this machine on http://{LOOPBACK}:{port}"),
    ]


def reload_cert(context: ssl.SSLContext, certfile: str, keyfile: str) -> bool:
    """Load the files into the live context; new handshakes use them."""
    try:
        context.load_cert_chain(certfile, keyfile)
        return True
    except (OSError, ssl.SSLError, ValueError) as exc:
        print(f"[https] could not reload the certificate (keeping the old one): {exc}",
              flush=True)
        return False


def _stamp(*paths: str):
    try:
        return tuple(os.stat(p).st_mtime_ns for p in paths)
    except OSError:
        return None


async def _watch_cert(context: ssl.SSLContext, certfile: str, keyfile: str,
                      stop: asyncio.Event, interval: float) -> None:
    seen = _stamp(certfile, keyfile)
    while not stop.is_set():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval)
        now = _stamp(certfile, keyfile)
        if now is not None and now != seen:
            seen = now
            if reload_cert(context, certfile, keyfile):
                print("[https] renewed certificate loaded", flush=True)


async def serve(servers, certfile: str, keyfile: str,
                stop: Optional[asyncio.Event] = None,
                cert_interval: float = CERT_CHECK_SECONDS) -> None:
    """Run every server until a signal (or `stop`) ends them all."""
    stop = stop or asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_signal():
        if stop.is_set():
            for server, _, _ in servers:
                server.force_exit = True
        stop.set()

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, on_signal)

    async def stopper():
        await stop.wait()
        for server, _, _ in servers:
            server.should_exit = True

    tasks = [asyncio.create_task(server.serve(sockets=socks)) for server, socks, _ in servers]
    helpers = [asyncio.create_task(stopper())]

    # The TLS context exists once the first server has loaded its config.
    tls = servers[0][0]
    while not tls.started and not any(t.done() for t in tasks):
        await asyncio.sleep(0.05)
    if tls.config.ssl is not None:
        helpers.append(asyncio.create_task(
            _watch_cert(tls.config.ssl, certfile, keyfile, stop, cert_interval)))
    for _, _, what in servers:
        print(f"[https] {what}", flush=True)

    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        stop.set()  # one ended (error or signal): end the others too
        await asyncio.gather(*tasks, return_exceptions=True)
        for t in done:
            if t.exception() is not None:
                raise t.exception()
    finally:
        stop.set()
        for h in helpers:
            h.cancel()
        await asyncio.gather(*helpers, return_exceptions=True)


def run(app, host: str, port: int, certfile: Optional[str] = None,
        keyfile: Optional[str] = None) -> None:
    """Serve `app`: plain HTTP as before, or HTTPS plus loopback HTTP."""
    if not (certfile and keyfile):
        uvicorn.run(app, host=host, port=port, reload=False)
        return
    servers = build(app, host, port, certfile, keyfile)
    asyncio.run(serve(servers, certfile, keyfile))
