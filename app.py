"""Entry point — run the OpenAI-compatible DeepSeek server.

    python app.py            # serves on http://localhost:8010

On first use (no saved session) a browser window opens for you to sign in by
hand; run `python -m deepseek.auth` to do that ahead of time. Set
DEEPSEEK_PROFILE_DIR to reuse an existing signed-in Chrome profile.
"""

import os

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from server.config import ENABLE_WAF_SIDECAR, WAF_REFRESH_INTERVAL
    if ENABLE_WAF_SIDECAR:
        try:
            from deepseek.auth import start_waf_sidecar
            start_waf_sidecar(interval_seconds=WAF_REFRESH_INTERVAL)
        except Exception as e:
            print(f"[warning] Could not start WAF sidecar: {e}")

    # Serve HTTPS when a cert pair is configured (./setup-https.sh gets one).
    # A browser page loaded over https:// cannot call an http:// API (mixed
    # content), so an internet-facing deployment needs TLS. This machine's own
    # tools keep plain HTTP on 127.0.0.1 at the same port; see server/listen.py.
    from server.listen import run

    run(
        "server.api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8010")),
        certfile=os.getenv("SSL_CERTFILE") or None,
        keyfile=os.getenv("SSL_KEYFILE") or None,
    )
