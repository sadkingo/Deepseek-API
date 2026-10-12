"""Register a fresh DeepSeek account through the Chrome extension.

The sign-up automation in ~/Desktop/Create account deepseek/register.py drives
a Chrome it launched itself. The extension in extension/ does the same steps
inside the Chrome you already run, which has no DevTools port, no automation
switches and nothing else that makes a browser look launched by a script.

An extension cannot listen on a socket or read IMAP, so this module is the
broker between whoever wants an account and the browser that can make one:

    POST /register                    -> queue a job (returns it)
    GET  /register                    -> status: current job, extension seen
    GET  /register/jobs/next          -> the extension long-polls for work
    GET  /register/jobs/{id}          -> job state and log, for callers
    POST /register/jobs/{id}/progress -> the extension reports each step
    GET  /register/jobs/{id}/code     -> long-poll: the 6-digit code from Gmail
    POST /register/jobs/{id}/result   -> {token, cookies, user_agent}: saved
                                         as session/session.json and installed
                                         in the running server
    POST /register/jobs/{id}/fail     -> {error}

The job carries the email and password to sign up with, so the routes answer
only direct, unproxied requests from this machine (not the Cloudflare tunnel,
not a LAN client, not a web page). REGISTER_TOKEN adds a shared secret on top.

The email, password and Gmail app password come from the sign-up project's
.env (ACCOUNT_ENV_FILE), exactly as register.py reads them, or from this
process's environment. `python -m server.register` triggers a job from the
command line and follows it to the end.
"""

from __future__ import annotations

import argparse
import email
import imaplib
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from deepseek.auth import (DEFAULT_ACCOUNT_ENV, DEFAULT_SESSION_FILE, Session,
                           _read_env_file, mask_email)

# The project's .env (PORT, REGISTER_TOKEN, ...) — also for the command line,
# which does not go through server/api.py.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# How long the extension may wait for the verification email after "Send code"
# worked (the same bound as register.py's CODE_WAIT_SECONDS).
CODE_WAIT_SECONDS = float(os.getenv("CODE_WAIT_SECONDS", "600"))
# A job the extension stopped reporting on for this long is given up on, so a
# service worker Chrome killed mid-run cannot leave a caller waiting forever.
JOB_STALE_SECONDS = float(os.getenv("REGISTER_STALE_SECONDS", "900"))
# Chrome ends an extension's fetch that gets no answer for 30 s; long polls
# must return before that.
LONG_POLL_MAX = 25.0
REGISTER_TOKEN = os.getenv("REGISTER_TOKEN") or None

SESSION_FILE = Path(os.getenv("TARGET_SESSION_FILE") or DEFAULT_SESSION_FILE)
ACCOUNT_ENV = DEFAULT_ACCOUNT_ENV

# Set by server/api.py: called with the new Session once it is saved, so the
# running server switches to the fresh account without a restart.
session_installed: Optional[Callable[[Session], None]] = None

socket.setdefaulttimeout(20)  # IMAP must never hang a request thread


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

class Credentials:
    """What a sign-up needs; environment first, then the sign-up project's .env."""

    def __init__(self, env_file: Path = None) -> None:
        env = _read_env_file(env_file or ACCOUNT_ENV)

        def pick(key: str) -> str:
            return os.getenv(key, "").strip() or env.get(key, "").strip()

        self.email = pick("DEEPSEEK_EMAIL")
        self.password = pick("DEEPSEEK_PASSWORD")
        self.gmail_email = pick("GMAIL_EMAIL") or self.email
        self.gmail_app_password = pick("GMAIL_APP_PASSWORD")

    @property
    def missing(self) -> List[str]:
        out = []
        if not self.email:
            out.append("DEEPSEEK_EMAIL")
        if not self.password:
            out.append("DEEPSEEK_PASSWORD")
        if not self.gmail_app_password:
            out.append("GMAIL_APP_PASSWORD")
        return out


def update_env_email(addr: str, env_file: Path = None) -> None:
    """Point the sign-up project's .env at the account that now exists, so the
    server's automatic re-login (deepseek.auth.relogin) signs into it."""
    path = env_file or ACCOUNT_ENV
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return
    out, seen = [], False
    for line in lines:
        if line.startswith("DEEPSEEK_EMAIL="):
            out.append(f"DEEPSEEK_EMAIL={addr}\n")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"DEEPSEEK_EMAIL={addr}\n")
    try:
        path.write_text("".join(out), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Gmail: the verification code, and which "+N" alias to sign up with
# ---------------------------------------------------------------------------

def parse_email(addr: str):
    """'user+41@gmail.com' -> ('user', 41); 'user@gmail.com' -> ('user', None)."""
    local, _, _domain = addr.partition("@")
    base, plus, suffix = local.partition("+")
    if plus and suffix.isdigit():
        return base, int(suffix)
    return base, None


def extract_code(text: str) -> Optional[str]:
    """The first 6-digit group in the human-readable part of a message."""
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    m = re.search(r"\b(\d{6})\b", text)
    return m.group(1) if m else None


def _msg_recipients(msg) -> set:
    recipients = set()
    for header in ("Delivered-To", "To", "X-Original-To", "Envelope-To"):
        for raw in msg.get_all(header, []):
            for addr in re.findall(r"[\w.+\-]+@[\w.\-]+", raw):
                recipients.add(addr.lower())
    return recipients


def _msg_text(msg) -> str:
    parts = []
    if msg.is_multipart():
        for p in msg.walk():
            if p.get_content_type() in ("text/plain", "text/html"):
                payload = p.get_payload(decode=True)
                if payload:
                    parts.append(payload.decode("utf-8", "ignore"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            parts.append(payload.decode("utf-8", "ignore"))
    return "\n".join(parts)


class Gmail:
    """The IMAP side of register.py, as one object per (user, app password)."""

    def __init__(self, user: str, app_password: str) -> None:
        self.user = user
        self.app_password = app_password

    def _connect(self):
        mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        mail.login(self.user, self.app_password)
        mail.select("INBOX")
        return mail

    def latest_uid(self) -> Optional[int]:
        """The newest UID now, so only mail after it counts as the answer."""
        mail = self._connect()
        try:
            status, data = mail.uid("search", None, "ALL")
            if status == "OK" and data and data[0]:
                return int(data[0].split()[-1])
            return None
        finally:
            self._logout(mail)

    def highest_registered_num(self) -> Optional[int]:
        """The highest '+N' any DeepSeek mail in the inbox was sent to."""
        mail = self._connect()
        highest = None
        try:
            _status, data = mail.uid("search", None, 'FROM "deepseek@"')
            for num in (data[0].split() if data and data[0] else []):
                status, msg_data = mail.uid(
                    "fetch", num, "(BODY.PEEK[HEADER.FIELDS (TO DELIVERED-TO)])")
                if status != "OK" or not msg_data or not msg_data[0]:
                    continue
                for part in msg_data:
                    if isinstance(part, tuple):
                        txt = part[1].decode("utf-8", "ignore")
                        for m in re.findall(r"[\w.\-]+\+(\d+)@", txt):
                            n = int(m)
                            if highest is None or n > highest:
                                highest = n
        finally:
            self._logout(mail)
        return highest

    def is_registered(self, addr: str) -> bool:
        """Whether this exact address has already received DeepSeek mail."""
        mail = self._connect()
        try:
            _status, data = mail.uid("search", None, f'FROM "deepseek@" TO "{addr}"')
            return bool(data and data[0] and data[0].split())
        finally:
            self._logout(mail)

    def resolve_next_email(self, addr: str) -> str:
        """`addr` if it is still free, else base+(highest N + 1)@domain."""
        base, _ = parse_email(addr)
        domain = addr.partition("@")[2] or "gmail.com"
        if self.is_registered(addr):
            return f"{base}+{(self.highest_registered_num() or 0) + 1}@{domain}"
        return addr

    def wait_for_code(self, after_uid: Optional[int], for_address: str,
                      deadline: float, stop: threading.Event,
                      on_status: Callable[[str], None] = lambda s: None,
                      ) -> Optional[str]:
        """Poll the inbox until a 6-digit code addressed to `for_address`
        arrives in a message newer than `after_uid`, or the deadline passes."""
        criteria = f"UID {int(after_uid) + 1}:*" if after_uid is not None else "ALL"
        mail = self._connect()
        on_status("connected to Gmail, waiting for the verification email")
        try:
            while not stop.is_set() and time.time() < deadline:
                try:
                    mail.noop()
                    _status, data = mail.uid("search", None, criteria)
                    for num in reversed(data[0].split() if data and data[0] else []):
                        status, msg_data = mail.uid("fetch", num, "(BODY.PEEK[])")
                        if status != "OK" or not msg_data or not msg_data[0]:
                            continue
                        for part in msg_data:
                            if not isinstance(part, tuple):
                                continue
                            msg = email.message_from_bytes(part[1])
                            if for_address.lower() not in _msg_recipients(msg):
                                continue
                            code = extract_code(_msg_text(msg))
                            if code:
                                return code
                except imaplib.IMAP4.abort:
                    on_status("Gmail connection dropped, reconnecting")
                    self._logout(mail)
                    mail = self._connect()
                except Exception as exc:  # transient IMAP trouble
                    on_status(f"Gmail: {exc}")
                stop.wait(1)
        finally:
            self._logout(mail)
        return None

    @staticmethod
    def _logout(mail) -> None:
        try:
            mail.logout()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, email_addr: str, password: str, reason: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.email = email_addr
        self.password = password
        self.reason = reason
        self.state = "pending"     # pending -> running -> done | failed
        self.stage = "queued"
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.finished_at: Optional[float] = None
        self.log: List[dict] = []
        self.baseline_uid: Optional[int] = None
        self.code: Optional[str] = None
        self.code_error: Optional[str] = None
        self._code_thread: Optional[threading.Thread] = None
        # Stops the Gmail watcher of the current "Send code" click.
        self._watch_stop: Optional[threading.Event] = None
        self.note(f"queued ({reason}) for {mask_email(email_addr)}")

    def stop_watch(self) -> None:
        if self._watch_stop is not None:
            self._watch_stop.set()

    def note(self, message: str, stage: Optional[str] = None) -> None:
        if stage:
            self.stage = stage
        self.updated_at = time.time()
        self.log.append({"t": self.updated_at, "stage": self.stage, "message": message})

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed")

    def as_dict(self, secrets: bool = False) -> dict:
        out = {
            "id": self.id, "state": self.state, "stage": self.stage,
            "email": mask_email(self.email), "reason": self.reason,
            "error": self.error, "created_at": self.created_at,
            "updated_at": self.updated_at, "finished_at": self.finished_at,
            "log": list(self.log),
        }
        if secrets:
            out["email"] = self.email
            out["password"] = self.password
        return out


class Broker:
    """One job at a time; the extension pulls it, the server feeds the code."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.jobs: Dict[str, Job] = {}
        self.current: Optional[Job] = None
        self.extension_seen_at: Optional[float] = None

    # -- creation ---------------------------------------------------------
    def request(self, reason: str = "requested") -> Job:
        """Queue a job, or hand back the one already in flight."""
        with self._lock:
            if self.current is not None and not self.current.finished:
                self._expire_locked(self.current)
                if not self.current.finished:
                    return self.current
        creds = Credentials()
        if creds.missing:
            raise LookupError("sign-up credentials missing: " + ", ".join(creds.missing)
                              + f" (set them in the environment or in {ACCOUNT_ENV})")
        gmail = Gmail(creds.gmail_email, creds.gmail_app_password)
        address = gmail.resolve_next_email(creds.email)
        job = Job(address, creds.password, reason)
        if address != creds.email:
            job.note(f"{mask_email(creds.email)} is already registered; using "
                     f"{mask_email(address)}")
        with self._cond:
            self.jobs[job.id] = job
            self.current = job
            self._cond.notify_all()
        return job

    # -- the extension's side ----------------------------------------------
    def next_pending(self, wait: float) -> Optional[Job]:
        deadline = time.time() + max(0.0, min(wait, LONG_POLL_MAX))
        with self._cond:
            self.extension_seen_at = time.time()
            while True:
                job = self.current
                if job is not None:
                    self._expire_locked(job)
                    if job.state == "pending":
                        job.state = "running"
                        job.note("picked up by the extension", stage="claimed")
                        self._cond.notify_all()
                        return job
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is not None:
                self._expire_locked(job)
            return job

    def progress(self, job: Job, stage: str, message: str) -> None:
        with self._cond:
            if job.finished:
                return
            job.note(message or stage, stage=stage)
            self._cond.notify_all()
        if stage == "send_code":
            # The code that answers THIS click is the only one that counts:
            # remember where the inbox ends right now.
            creds = Credentials()
            try:
                uid = Gmail(creds.gmail_email, creds.gmail_app_password).latest_uid()
            except Exception as exc:
                uid = None
                self._say(job, f"could not read the inbox baseline: {exc}")
            with self._cond:
                job.baseline_uid = uid
                # A fresh click means a fresh code: forget any earlier one
                # and the watcher that was looking for it.
                job.stop_watch()
                job._watch_stop = None
                job._code_thread = None
                job.code, job.code_error = None, None
                self._cond.notify_all()

    def wait_code(self, job: Job, wait: float) -> dict:
        """Start watching Gmail (once) and wait up to `wait` for the code."""
        with self._cond:
            if job.finished:
                return {"error": job.error or "job is over"}
            if job._code_thread is None:
                stop = threading.Event()
                job._watch_stop = stop
                job._code_thread = threading.Thread(
                    target=self._watch_code, args=(job, stop), daemon=True,
                    name=f"gmail-code-{job.id}")
                job._code_thread.start()
            deadline = time.time() + max(0.0, min(wait, LONG_POLL_MAX))
            while job.code is None and job.code_error is None and not job.finished:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return {"pending": True}
                self._cond.wait(remaining)
            if job.code is not None:
                return {"code": job.code}
            return {"error": job.code_error or job.error or "job is over"}

    def _watch_code(self, job: Job, stop: threading.Event) -> None:
        creds = Credentials()
        try:
            code = Gmail(creds.gmail_email, creds.gmail_app_password).wait_for_code(
                job.baseline_uid, job.email, time.time() + CODE_WAIT_SECONDS,
                stop, on_status=lambda s: self._say(job, s))
            error = None if code else f"no verification code within {CODE_WAIT_SECONDS:.0f}s"
        except Exception as exc:
            code, error = None, f"Gmail: {exc}"
        with self._cond:
            if job._watch_stop is not stop:
                return  # superseded by a later "Send code"
            job.code, job.code_error = code, error
            job.note("verification code found" if code else error)
            self._cond.notify_all()

    def finish(self, job: Job, payload: dict) -> Session:
        token = str(payload.get("token") or "")
        cookies = payload.get("cookies") or {}
        if not isinstance(cookies, dict):
            raise ValueError("cookies must be an object of name -> value")
        cookies = {str(k): str(v) for k, v in cookies.items()}
        if not token and not cookies:
            raise ValueError("nothing captured (no token, no cookies)")
        session = Session(token=token, cookies=cookies,
                          user_agent=str(payload.get("user_agent") or ""),
                          captured_at=time.time())
        session.save(SESSION_FILE)
        update_env_email(job.email)
        with self._cond:
            job.state, job.finished_at = "done", time.time()
            job.stop_watch()
            job.note(f"session saved to {SESSION_FILE} "
                     f"(token={'yes' if token else 'NO'}, {len(cookies)} cookies)",
                     stage="done")
            self._cond.notify_all()
        if session_installed is not None:
            try:
                session_installed(session)
                self._say(job, "server switched to the new account")
            except Exception as exc:
                self._say(job, f"session saved, but the server could not switch to it: {exc}")
        return session

    def fail(self, job: Job, error: str) -> None:
        with self._cond:
            if job.finished:
                return
            job.state, job.error, job.finished_at = "failed", error, time.time()
            job.stop_watch()
            job.note(error, stage="failed")
            self._cond.notify_all()

    def wait_done(self, job: Job, timeout: float) -> bool:
        deadline = time.time() + timeout
        with self._cond:
            while not job.finished:
                self._expire_locked(job)
                remaining = deadline - time.time()
                if remaining <= 0 or job.finished:
                    break
                self._cond.wait(min(remaining, 5.0))
            return job.finished

    # -- helpers -------------------------------------------------------------
    def _say(self, job: Job, message: str) -> None:
        with self._cond:
            job.note(message)
            self._cond.notify_all()

    def _expire_locked(self, job: Job) -> None:
        if job.finished or time.time() - job.updated_at < JOB_STALE_SECONDS:
            return
        job.state, job.finished_at = "failed", time.time()
        job.error = (f"no word from the extension for {JOB_STALE_SECONDS:.0f}s; "
                     "is Chrome open with the extension installed?")
        job.stop_watch()
        job.note(job.error, stage="failed")
        self._cond.notify_all()


broker = Broker()


# ---------------------------------------------------------------------------
# Getting Chrome (and with it the extension) on its feet
# ---------------------------------------------------------------------------

CHROME_CANDIDATES = ("google-chrome-stable", "google-chrome", "chromium",
                     "/opt/google/chrome/google-chrome",
                     "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
# How long a poll from the extension is taken as proof that Chrome is open.
EXTENSION_RECENT = 60.0


def find_chrome() -> Optional[str]:
    import shutil
    forced = os.getenv("CHROME_BINARY")
    if forced:
        return forced
    for name in CHROME_CANDIDATES:
        path = name if os.path.isabs(name) else shutil.which(name)
        if path and os.path.exists(path):
            return path
    return None


def launch_chrome() -> str:
    """Start the user's Chrome the way the desktop does, so the extension
    installed in it wakes up and polls. Returns what was run."""
    import subprocess
    binary = find_chrome()
    if not binary:
        raise FileNotFoundError("Google Chrome not found; set CHROME_BINARY")
    args = [binary]
    profile = os.getenv("CHROME_PROFILE_DIRECTORY")
    if profile:
        args.append(f"--profile-directory={profile}")
    args.append("--no-first-run")
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    return " ".join(args)


def ensure_extension(broker: "Broker", grace: float = 35.0,
                     say: Callable[[str], None] = lambda s: None) -> bool:
    """Make sure something will pick the job up.

    A poll within the last minute means Chrome is open and the extension is
    awake. Otherwise wait one poll interval (its alarm fires every 30 s) and,
    if it still has not called, start Chrome. Returns True when the extension
    polled, False when Chrome had to be launched (or could not be)."""
    def polled_recently() -> bool:
        seen = broker.extension_seen_at
        return bool(seen) and time.time() - seen < EXTENSION_RECENT

    if polled_recently():
        return True
    say(f"waiting up to {grace:.0f}s for the extension to poll")
    deadline = time.time() + grace
    while time.time() < deadline:
        if polled_recently():
            return True
        time.sleep(1)
    try:
        say(f"no poll from the extension; starting Chrome: {launch_chrome()}")
    except Exception as exc:
        say(f"could not start Chrome: {exc}")
    return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/register", tags=["register"])


def _guard(request: Request) -> Optional[JSONResponse]:
    """Only this machine, directly, and not from a web page.

    The Cloudflare tunnel and any reverse proxy connect from 127.0.0.1 too,
    but tag the request with forwarding headers; a page in a browser tab is
    loopback as well, but its fetch carries a http(s) Origin. Curl, Python and
    the extension's service worker carry neither.
    """
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "testclient"):
        return _deny("register is only available from this machine")
    for header in ("x-forwarded-for", "x-real-ip", "cf-connecting-ip", "forwarded"):
        if request.headers.get(header):
            return _deny("register is not available through a proxy or tunnel")
    origin = request.headers.get("origin", "")
    if origin and not origin.startswith(("chrome-extension://", "moz-extension://")):
        return _deny("register is not available to web pages")
    if REGISTER_TOKEN and request.headers.get("x-register-token") != REGISTER_TOKEN:
        return _deny("missing or wrong X-Register-Token", 401)
    return None


def _deny(message: str, status: int = 403) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def _job_or_404(job_id: str):
    job = broker.get(job_id)
    if job is None:
        return None, JSONResponse(status_code=404, content={"error": "no such job"})
    return job, None


@router.get("")
def status(request: Request):
    if (denied := _guard(request)) is not None:
        return denied
    creds = Credentials()
    current = broker.current
    return {
        "credentials_ok": not creds.missing,
        "credentials_missing": creds.missing,
        "account_env": str(ACCOUNT_ENV),
        "session_file": str(SESSION_FILE),
        "extension_seen_at": broker.extension_seen_at,
        "current": current.as_dict() if current else None,
    }


@router.post("")
async def create(request: Request):
    if (denied := _guard(request)) is not None:
        return denied
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}
    reason = str(body.get("reason") or "requested")
    try:
        job = await _threaded(broker.request, reason)
    except LookupError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"Gmail: {exc}"})
    wait = float(body.get("wait") or 0)
    if wait > 0:
        await _threaded(broker.wait_done, job, wait)
    return job.as_dict()


@router.get("/jobs/next")
def next_job(request: Request, wait: float = 20.0):
    if (denied := _guard(request)) is not None:
        return denied
    job = broker.next_pending(wait)
    if job is None:
        return Response(status_code=204)
    return job.as_dict(secrets=True)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    if (denied := _guard(request)) is not None:
        return denied
    job, missing = _job_or_404(job_id)
    return missing or job.as_dict()


@router.post("/jobs/{job_id}/progress")
async def progress(job_id: str, request: Request):
    if (denied := _guard(request)) is not None:
        return denied
    job, missing = _job_or_404(job_id)
    if missing:
        return missing
    body = await request.json()
    stage = str(body.get("stage") or "working")
    await _threaded(broker.progress, job, stage, str(body.get("message") or ""))
    return {"ok": True}


@router.get("/jobs/{job_id}/code")
def code(job_id: str, request: Request, wait: float = 20.0):
    if (denied := _guard(request)) is not None:
        return denied
    job, missing = _job_or_404(job_id)
    return missing or broker.wait_code(job, wait)


@router.post("/jobs/{job_id}/result")
async def result(job_id: str, request: Request):
    if (denied := _guard(request)) is not None:
        return denied
    job, missing = _job_or_404(job_id)
    if missing:
        return missing
    body = await request.json()
    try:
        await _threaded(broker.finish, job, body)
    except ValueError as exc:
        broker.fail(job, str(exc))
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return job.as_dict()


@router.post("/jobs/{job_id}/fail")
async def fail(job_id: str, request: Request):
    if (denied := _guard(request)) is not None:
        return denied
    job, missing = _job_or_404(job_id)
    if missing:
        return missing
    body = await request.json()
    broker.fail(job, str(body.get("error") or "the extension gave up"))
    return job.as_dict()


async def _threaded(fn, *args):
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(fn, *args)


# ---------------------------------------------------------------------------
# Command line: `python -m server.register` (bin/register)
# ---------------------------------------------------------------------------

class Unreachable(Exception):
    pass


def _http(method: str, url: str, body: Optional[dict] = None, timeout: float = 30) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    if REGISTER_TOKEN:
        req.add_header("X-Register-Token", REGISTER_TOKEN)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read()).get("error") or exc.reason
        except Exception:
            message = exc.reason
        raise SystemExit(f"{method} {url} -> HTTP {exc.code}: {message}")
    except urllib.error.URLError as exc:
        raise Unreachable(f"cannot reach the server at {url}: {exc.reason}")
    return json.loads(raw) if raw else {}


def serve_broker(host: str, port: int, wait: float = 15.0):
    """Host just the /register routes, for when the API server is not running.

    The extension polls one address; whoever listens there is the broker. The
    session file this writes is what the API server loads when it starts.
    Returns the uvicorn server (set `should_exit` to stop it).
    """
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="DeepSeek register broker")
    app.include_router(router)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, name="register-broker", daemon=True).start()
    deadline = time.time() + wait
    while time.time() < deadline:
        if server.started:
            return server
        if not server.should_exit and not server.started:
            time.sleep(0.1)
            continue
        break
    raise SystemExit(f"could not listen on {host}:{port}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="register",
        description="Register a fresh DeepSeek account through the Chrome "
                    "extension and follow it to the end. Uses the running API "
                    "server, or hosts the /register routes itself if none is "
                    "up; starts Chrome if the extension is not polling.")
    parser.add_argument("--server", default=os.getenv("REGISTER_SERVER")
                        or f"http://127.0.0.1:{os.getenv('PORT', '8010')}")
    parser.add_argument("--timeout", type=float, default=CODE_WAIT_SECONDS + 300,
                        help="give up after this many seconds (default %(default)s)")
    parser.add_argument("--no-wait", action="store_true",
                        help="queue the job and exit without following it")
    args = parser.parse_args(argv)
    base = args.server.rstrip("/")

    # No server running? Then be the broker for this one job: the extension
    # polls this address regardless of who answers.
    own_broker = None
    try:
        info = _http("GET", f"{base}/register")
    except Unreachable as exc:
        from urllib.parse import urlparse
        url = urlparse(base)
        if url.hostname not in ("127.0.0.1", "localhost", "::1"):
            print(exc, file=sys.stderr)
            return 1
        if args.no_wait:
            print(f"{exc}\n--no-wait needs a running server to hold the job; "
                  "start it (`start`) or drop --no-wait.", file=sys.stderr)
            return 1
        print(f"The API server is not running at {base}; hosting the register "
              "routes here until the account is made.", flush=True)
        own_broker = serve_broker(url.hostname, url.port or 80)
        info = _http("GET", f"{base}/register")

    try:
        if not info["credentials_ok"]:
            print("Sign-up credentials missing: " + ", ".join(info["credentials_missing"])
                  + f"\nSet them in {info['account_env']} (or the environment).",
                  file=sys.stderr)
            return 1
        seen = info.get("extension_seen_at")
        if own_broker is not None:
            print("Waiting for the extension's next poll (it checks every 30 s; "
                  "Chrome must be open with the extension loaded)…", flush=True)
        elif not seen or time.time() - seen > 90:
            print("Note: the extension has not polled the server in the last 90s. "
                  "Is Chrome open with it installed and pointed at this server?",
                  file=sys.stderr)

        job = _http("POST", f"{base}/register", {"reason": "register command"})
        print(f"job {job['id']}: {job['stage']} ({job['email']})")
        if args.no_wait:
            return 0
        if own_broker is not None:
            # Nobody else will get Chrome going; if the extension does not
            # call within a poll interval, start Chrome ourselves.
            threading.Thread(
                target=ensure_extension, args=(broker,),
                kwargs={"say": lambda s: print(f"[chrome] {s}", flush=True)},
                daemon=True).start()

        shown = 0
        deadline = time.time() + args.timeout
        while True:
            job = _http("GET", f"{base}/register/jobs/{job['id']}")
            for entry in job["log"][shown:]:
                stamp = time.strftime("%H:%M:%S", time.localtime(entry["t"]))
                print(f"[{stamp}] {entry['stage']}: {entry['message']}", flush=True)
            shown = len(job["log"])
            if job["state"] == "done":
                if own_broker is not None:
                    print("Registered; the session is saved for the API server "
                          "to use when it starts.")
                else:
                    print("Registered; the server is using the new account. "
                          "Restart nothing.")
                return 0
            if job["state"] == "failed":
                print(f"Registration failed: {job['error']}", file=sys.stderr)
                return 1
            if time.time() > deadline:
                print(f"Gave up waiting after {args.timeout:.0f}s (job {job['id']} "
                      f"is still {job['state']}).", file=sys.stderr)
                return 1
            time.sleep(1)
    except Unreachable as exc:
        print(f"{exc} (the server went away mid-job)", file=sys.stderr)
        return 1
    finally:
        if own_broker is not None:
            own_broker.should_exit = True


if __name__ == "__main__":
    sys.exit(main())
