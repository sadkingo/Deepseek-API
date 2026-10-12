"""server/listen.py: HTTPS to the network, plain HTTP to this machine, one port.

Uses throwaway self-signed certificates. 127.0.0.2 stands in for "the
network": it is not 127.0.0.1, so it lands on the wildcard (TLS) socket the
same way a connection to the LAN address or the public domain does.
"""

import asyncio
import json
import os
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from server import listen


async def tiny_app(scope, receive, send):
    if scope["type"] != "http":
        return
    body = json.dumps({"scheme": scope["scheme"], "client": scope["client"][0]}).encode()
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": body})


def make_cert(directory: Path, name: str):
    cert, key = directory / f"{name}.crt", directory / f"{name}.key"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(key), "-out", str(cert), "-days", "1",
                    "-subj", f"/CN={name}"], check=True, capture_output=True)
    return str(cert), str(key)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def served_cert_cn(host: str, port: int) -> str:
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    with socket.create_connection((host, port), timeout=5) as raw:
        with ctx.wrap_socket(raw) as tls:
            der = tls.getpeercert(binary_form=True)
    pem = ssl.DER_cert_to_PEM_cert(der)
    out = subprocess.run(["openssl", "x509", "-noout", "-subject"], input=pem,
                         capture_output=True, text=True, check=True).stdout
    return out.strip().rsplit("=", 1)[-1].strip()


@unittest.skipUnless(shutil.which("openssl") and hasattr(socket, "SO_REUSEPORT"),
                     "needs openssl and SO_REUSEPORT")
class SplitListener(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        a_cert, a_key = make_cert(d, "first")
        self.certfile, self.keyfile = str(d / "live.crt"), str(d / "live.key")
        shutil.copy(a_cert, self.certfile)
        shutil.copy(a_key, self.keyfile)
        self.second = make_cert(d, "second")
        self.port = free_port()
        self.stop = None
        self.loop = None

    def start(self, **kw):
        servers = listen.build(tiny_app, "0.0.0.0", self.port, self.certfile,
                               self.keyfile, log_level="warning", **kw)
        ready = threading.Event()

        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.stop = asyncio.Event()
            ready.set()
            self.loop.run_until_complete(listen.serve(
                servers, self.certfile, self.keyfile, stop=self.stop, cert_interval=0.2))
            self.loop.close()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        ready.wait(5)
        deadline = time.time() + 10
        while time.time() < deadline and not all(s.started for s, _, _ in servers):
            time.sleep(0.05)
        self.addCleanup(self.shutdown)
        return servers

    def shutdown(self):
        if self.loop and self.stop:
            self.loop.call_soon_threadsafe(self.stop.set)
            self.thread.join(10)

    def get(self, url):
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        with urllib.request.urlopen(url, timeout=5, context=ctx) as r:
            return json.loads(r.read())

    def test_loopback_is_plain_and_the_rest_is_tls(self):
        self.start()
        self.assertEqual(self.get(f"http://127.0.0.1:{self.port}/")["scheme"], "http")
        self.assertEqual(self.get(f"https://127.0.0.2:{self.port}/")["scheme"], "https")
        # No plain HTTP from the network side, and no TLS on loopback.
        with self.assertRaises(Exception):
            self.get(f"http://127.0.0.2:{self.port}/")
        with self.assertRaises(Exception):
            self.get(f"https://127.0.0.1:{self.port}/")

    def test_local_http_off_means_tls_everywhere(self):
        self.start(local_http=False)
        self.assertEqual(self.get(f"https://127.0.0.1:{self.port}/")["scheme"], "https")
        with self.assertRaises(Exception):
            self.get(f"http://127.0.0.1:{self.port}/")

    def test_a_second_server_on_the_same_port_is_refused(self):
        self.start()
        with self.assertRaises(SystemExit) as ctx:
            listen.build(tiny_app, "0.0.0.0", self.port, self.certfile, self.keyfile)
        self.assertIn("already in use", str(ctx.exception))

    def test_renewed_certificate_is_picked_up_without_a_restart(self):
        self.start()
        self.assertEqual(served_cert_cn("127.0.0.2", self.port), "first")
        time.sleep(0.05)
        shutil.copy(self.second[0], self.certfile)
        shutil.copy(self.second[1], self.keyfile)
        deadline = time.time() + 5
        cn = None
        while time.time() < deadline:
            cn = served_cert_cn("127.0.0.2", self.port)
            if cn == "second":
                break
            time.sleep(0.1)
        self.assertEqual(cn, "second")
        # The loopback side is still plain HTTP afterwards.
        self.assertEqual(self.get(f"http://127.0.0.1:{self.port}/")["scheme"], "http")


@unittest.skipUnless(shutil.which("openssl"), "needs openssl")
class CertChecks(unittest.TestCase):
    def test_missing_and_mismatched_files_are_reported(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as ctx:
                listen.check_cert_files(f"{d}/nope.crt", f"{d}/nope.key")
            self.assertIn("setup-https.sh", str(ctx.exception))
            a = make_cert(Path(d), "a")
            b = make_cert(Path(d), "b")
            with self.assertRaises(SystemExit) as ctx:
                listen.check_cert_files(a[0], b[1])
            self.assertIn("valid pair", str(ctx.exception))
            listen.check_cert_files(*a)  # a matching pair passes


if __name__ == "__main__":
    unittest.main()
