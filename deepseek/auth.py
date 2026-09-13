"""
Authentication — Playwright login + session capture.

Mirrors the Windows-Copilot-API design: the browser is used ONLY to establish a
signed-in session (handling the AWS WAF / "verify you're human" check and the
email/password form). It does not chat. We then capture the bearer token from
`localStorage.userToken` plus the session cookies, and hand them to the
pure-HTTP client in `deepseek.client`.

A persistent Chromium profile means the human-check is a one-time thing: once
you've signed in, later runs reuse the profile and capture the token headlessly.

    from deepseek.auth import get_session
    session = get_session()          # logs in (visible) the first time, else headless
    print(session.token[:8], "...")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

from playwright.sync_api import sync_playwright

_auth_log = logging.getLogger("deepseek.auth")

ROOT = Path(__file__).resolve().parent.parent
# Override with DEEPSEEK_PROFILE_DIR to reuse an existing signed-in Chrome profile.
DEFAULT_PROFILE_DIR = Path(os.getenv("DEEPSEEK_PROFILE_DIR", ROOT / "session" / "profile"))
DEFAULT_SESSION_FILE = ROOT / "session" / "session.json"

CHAT_URL = "https://chat.deepseek.com/"
SIGNIN_URL = "https://chat.deepseek.com/sign_in"

# Where the LAST registered account's email/password live, for signing back in
# automatically when DeepSeek invalidates the token. DEEPSEEK_EMAIL /
# DEEPSEEK_PASSWORD in this process's environment win; otherwise they are read
# from the sign-up automation's .env, which it keeps pointed at the newest
# account (override the path with ACCOUNT_ENV_FILE).
DEFAULT_ACCOUNT_ENV = Path(os.getenv(
    "ACCOUNT_ENV_FILE",
    Path.home() / "Desktop" / "Create account deepseek" / ".env"))
# How long a visible login window waits for a token before giving up.
LOGIN_TIMEOUT = int(os.getenv("LOGIN_TIMEOUT", "300"))

LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]
# Token is trusted for this long before we refresh it from the browser again.
SESSION_MAX_AGE = 6 * 60 * 60  # 6 hours


class LoginRequired(RuntimeError):
    """Raised when no usable session exists and interactive login is disallowed
    (e.g. inside the server, where we can't pop open a browser mid-request).
    The message tells the user how to log in."""

    DEFAULT = (
        "No DeepSeek session found. Log in first by running:\n"
        "    python -m deepseek.auth\n"
        "This opens a browser once so you can sign in and clear the human-check; "
        "afterwards the server reuses the saved session automatically."
    )

    def __init__(self, message: str = DEFAULT):
        super().__init__(message)


@dataclass
class Session:
    """A captured signed-in DeepSeek session."""

    token: str
    cookies: Dict[str, str]
    user_agent: str
    captured_at: float

    @property
    def age(self) -> float:
        return time.time() - self.captured_at

    def save(self, path: Path = DEFAULT_SESSION_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_SESSION_FILE) -> Optional["Session"]:
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None


# --- account credentials ------------------------------------------------------

def _read_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def load_account_credentials(env_file: Path = DEFAULT_ACCOUNT_ENV) -> Optional[tuple]:
    """(email, password) of the account to sign in with, or None if unknown."""
    email = os.getenv("DEEPSEEK_EMAIL", "").strip()
    password = os.getenv("DEEPSEEK_PASSWORD", "")
    if not (email and password):
        env = _read_env_file(env_file)
        email = email or env.get("DEEPSEEK_EMAIL", "").strip()
        password = password or env.get("DEEPSEEK_PASSWORD", "")
    if email and password:
        return email, password
    return None


def mask_email(email: str) -> str:
    user, _, domain = email.partition("@")
    return f"{user[:2]}***@{domain}" if domain else "***"


# --- in-page helpers --------------------------------------------------------

# Reads the bearer token the web app stores after login. Shape:
#   localStorage.userToken = {"value":"<TOKEN>","__version":"0"}
_READ_TOKEN_JS = """
() => {
  try {
    const raw = window.localStorage.getItem('userToken');
    if (!raw) return null;
    const o = JSON.parse(raw);
    return (o && o.value) ? o.value : null;
  } catch (e) { return null; }
}
"""


def _safe_evaluate(page, js: str):
    """Run page.evaluate, swallowing the benign 'Execution context was destroyed'
    error that fires when a navigation (e.g. the post-login redirect) happens to
    land mid-evaluate. Returns None on any such transient failure instead of
    raising, so callers can just retry on the next poll."""
    try:
        return page.evaluate(js)
    except Exception as e:
        msg = str(e)
        if "Execution context was destroyed" in msg or "navigation" in msg.lower():
            return None
        raise


def _capture_from_context(context, page) -> Optional[Session]:
    """Read token + cookies + UA off a logged-in page, or None if not signed in."""
    token = _safe_evaluate(page, _READ_TOKEN_JS)
    if not token:
        return None
    cookies = {c["name"]: c["value"] for c in context.cookies()}
    ua = _safe_evaluate(page, "() => navigator.userAgent") or ""
    return Session(token=token, cookies=cookies, user_agent=ua, captured_at=time.time())


def _wait_for_token(page, timeout: float) -> Optional[str]:
    """Poll localStorage.userToken until it appears or we time out. Tolerates
    transient navigations (e.g. the Google OAuth redirect chain) that briefly
    destroy the page's JS execution context — those just count as "no token
    yet" rather than aborting the whole login."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        token = _safe_evaluate(page, _READ_TOKEN_JS)
        if token:
            return token
        page.wait_for_timeout(1000)
    return None


def _first_visible(page, selectors):
    """The first selector that matches a visible element, or None."""
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible():
                return loc
        except Exception:
            continue
    return None


_EMAIL_FIELDS = (
    'input[placeholder*="email" i]', 'input[placeholder*="Email" i]',
    'input[type="email"]', 'input[type="text"]:not([type="password"])',
)
_PASSWORD_FIELDS = ('input[type="password"]', 'input[placeholder*="assword"]')
# The primary "Log in" first: "Log in with Google" contains the same text.
_LOGIN_BUTTONS = (
    '.ds-button--primary:has-text("Log in")',
    'div[role="button"]:has-text("Log in")', '.ds-button:has-text("Log in")',
    'button:has-text("Log in")', 'div[role="button"]:has-text("Sign in")',
    'button:has-text("Sign in")', 'div[role="button"]:has-text("登录")',
)


def _autofill_login(page, email: str, password: str) -> bool:
    """Best effort: type the credentials into the sign-in form and submit.

    Returns True if the form was submitted. Anything that does not look as
    expected just leaves the window for the user to finish by hand; the token
    poll afterwards is what decides success.
    """
    try:
        email_box = _first_visible(page, _EMAIL_FIELDS)
        pw_box = _first_visible(page, _PASSWORD_FIELDS)
        if email_box is None or pw_box is None:
            return False
        email_box.click()
        email_box.fill(email)
        pw_box.click()
        pw_box.fill(password)
        # The terms checkbox, if it is a real one and unticked.
        try:
            box = page.locator('input[type="checkbox"]').first
            if box.count() and not box.is_checked():
                box.check(force=True)
        except Exception:
            pass
        button = _first_visible(page, _LOGIN_BUTTONS)
        if button is not None:
            button.click()
        else:
            pw_box.press("Enter")
        return True
    except Exception as e:
        print(f"[auth] could not fill the sign-in form automatically ({e}); "
              "please sign in in the window.")
        return False


def _safe_goto(page, url: str) -> None:
    """Navigate, tolerating the benign `net::ERR_ABORTED` that DeepSeek's SPA
    redirects and the AWS WAF check often raise mid-navigation. We wait only for
    the initial commit, not full `load`; the token-poll afterwards is what
    actually gates sign-in, so an aborted/partial load here is fine."""
    try:
        page.goto(url, wait_until="commit", timeout=60000)
    except Exception as e:
        print(f"[auth] navigation to {url} was interrupted ({type(e).__name__}); "
              "continuing — finish signing in in the window if needed.")
    # Give the SPA a moment to render its login UI before we touch the form.
    page.wait_for_timeout(2000)


def login(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    headless: bool = False,
    assume_logged_out: bool = False,
    credentials: Optional[tuple] = None,
) -> Session:
    """Interactive login. Opens a visible window and waits for you to sign in by
    hand (and clear the AWS WAF human-check); once a token appears it captures
    and saves the session. The persistent profile means later `get_session()`
    calls capture the token headlessly without a window.

    With `credentials` — (email, password), defaulting to the last registered
    account (`load_account_credentials`) — the form is filled in and submitted
    automatically; the window is still shown so a human-check can be solved.

    `assume_logged_out=True` skips the initial "are we already signed in?" hop to
    CHAT_URL and goes straight to the sign-in page. Callers that have just
    confirmed there's no token (e.g. get_session after a failed headless refresh)
    pass this so the window doesn't visibly bounce CHAT_URL -> SIGNIN_URL."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(
                str(profile_dir), headless=headless, channel="chrome", args=LAUNCH_ARGS,
            )
        except Exception:
            context = p.chromium.launch_persistent_context(
                str(profile_dir), headless=headless, args=LAUNCH_ARGS,
            )
        page = context.pages[0] if context.pages else context.new_page()

        # Normally we first land on CHAT_URL to reuse an already-signed-in
        # profile. When the caller already knows we're logged out, skip straight
        # to the sign-in page so the window doesn't appear to "refresh".
        existing = None
        if not assume_logged_out:
            _safe_goto(page, CHAT_URL)
            existing = page.evaluate(_READ_TOKEN_JS)

        if not existing:
            _safe_goto(page, SIGNIN_URL)
            if credentials is None:
                credentials = load_account_credentials()
            if credentials and _autofill_login(page, *credentials):
                print(f"[auth] Signing in as {mask_email(credentials[0])} "
                      "automatically — solve the human-check in the window if "
                      "one appears. Waiting for the session...")
            else:
                print("[auth] Please sign in in the window (solve the human-check "
                      "if shown). Waiting for the session...")
            if not _wait_for_token(page, timeout=LOGIN_TIMEOUT):
                context.close()
                raise RuntimeError("Login timed out — no token captured.")

        session = _capture_from_context(context, page)
        context.close()
        if session is None:
            raise RuntimeError("Logged in but could not read the token.")
        session.save()
        return session


def _headless_refresh(profile_dir: Path) -> Optional[Session]:
    """Try to capture a token headlessly from the persistent profile. Returns a
    saved Session if the profile is still signed in, else None. Never opens a
    visible window."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(
                str(profile_dir), headless=True, channel="chrome", args=LAUNCH_ARGS,
            )
        except Exception:
            context = p.chromium.launch_persistent_context(
                str(profile_dir), headless=True, args=LAUNCH_ARGS,
            )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            _safe_goto(page, CHAT_URL)
            session = _capture_from_context(context, page)
        finally:
            context.close()

    if session is not None:
        session.save()
    return session


def get_session(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    session_file: Path = DEFAULT_SESSION_FILE,
    max_age: int = SESSION_MAX_AGE,
    allow_interactive: bool = True,
    force_refresh: bool = False,
) -> Session:
    """Return a usable session: cached file if fresh, else a headless refresh
    from the browser profile.

    If neither works and `allow_interactive` is True, open a visible window for
    manual sign-in. If it's False (the server's case — we can't pop a browser
    mid-request), raise `LoginRequired` telling the user to run the login step.

    Note: this uses Playwright's *sync* API, so it must not be called from inside
    an asyncio event loop — call it from a worker thread (e.g. run_in_threadpool)."""
    if not force_refresh:
        cached = Session.load(session_file)
        if cached and cached.age < max_age:
            return cached

    # Try a headless refresh from the (presumably logged-in) persistent profile.
    session = _headless_refresh(profile_dir)
    if session is not None:
        return session

    if not allow_interactive:
        raise LoginRequired()

    # Not logged in yet — open a visible window so the user can sign in (and
    # clear the human-check) by hand. The persistent profile means this only
    # happens once — later calls capture the token headlessly. We just confirmed
    # (above) there's no token, so go straight to the sign-in page.
    print("[auth] No valid session found — opening a browser window to log in...")
    return login(profile_dir=profile_dir, assume_logged_out=True)


def relogin(
    bad_token: Optional[str] = None,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    allow_interactive: bool = True,
) -> Session:
    """Get a NEW session after DeepSeek rejected the current token.

    First a headless capture from the profile, in case the browser session is
    still alive and merely holds a newer token; a capture that hands back the
    very token that was just rejected counts as signed out. Then, if allowed,
    a visible window signed in automatically with the last account's
    credentials (see `login`). Raises `LoginRequired` when interactive login
    is disallowed and nothing else worked.
    """
    session = _headless_refresh(profile_dir)
    if session is not None and session.token != bad_token:
        return session
    if not allow_interactive:
        raise LoginRequired(
            "DeepSeek rejected the saved token and no signed-in browser "
            "profile was found. Log in again with:\n    python -m deepseek.auth")
    print("[auth] DeepSeek rejected the token — opening a browser window to "
          "sign in again...")
    return login(profile_dir=profile_dir, assume_logged_out=True)


_waf_sidecar_thread: Optional[threading.Thread] = None


def start_waf_sidecar(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    interval_seconds: int = 2400,
) -> None:
    """Launch a background daemon thread that periodically refreshes the session
    and rotates AWS WAF tokens using the headless Playwright profile."""
    global _waf_sidecar_thread
    if _waf_sidecar_thread and _waf_sidecar_thread.is_alive():
        return

    def _worker():
        _auth_log.info("WAF sidecar started (refresh interval: %ds)", interval_seconds)
        while True:
            time.sleep(interval_seconds)
            try:
                _auth_log.debug("WAF sidecar: performing background session/WAF refresh...")
                s = _headless_refresh(profile_dir)
                if s and "aws-waf-token" in s.cookies:
                    _auth_log.info("WAF sidecar: rotated aws-waf-token successfully")
                elif s:
                    _auth_log.info("WAF sidecar: session refreshed successfully")
                else:
                    _auth_log.warning("WAF sidecar: headless refresh returned no session")
            except Exception as e:
                _auth_log.warning("WAF sidecar refresh error: %s", e)

    _waf_sidecar_thread = threading.Thread(target=_worker, name="waf-sidecar", daemon=True)
    _waf_sidecar_thread.start()


if __name__ == "__main__":
    s = login()
    print(f"[auth] captured token {s.token[:10]}... ({len(s.cookies)} cookies)")
