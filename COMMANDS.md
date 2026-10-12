# Commands

Four shortcuts do the everyday jobs. They work from any directory — each one
finds the project itself — and are defined as zsh functions in `~/.zshrc`
pointing at the scripts in [bin/](bin/):

| Command    | What it does                                        |
| ---------- | --------------------------------------------------- |
| `start`    | Run the API server (foreground; Ctrl-C stops it)    |
| `login`    | Sign in to DeepSeek in a browser and save the session |
| `register` | Create a fresh DeepSeek account through the Chrome extension |
| `tunnel`   | Expose the local server on a public HTTPS URL       |

To set them up on another machine, add this to `~/.zshrc` (or `~/.bashrc`) and
open a new terminal:

```bash
export DEEPSEEK_API_DIR="$HOME/Desktop/Deepseek-API"
start()    { "$DEEPSEEK_API_DIR/bin/start"    "$@" }
login()    { "$DEEPSEEK_API_DIR/bin/login"    "$@" }
register() { "$DEEPSEEK_API_DIR/bin/register" "$@" }
tunnel()   { "$DEEPSEEK_API_DIR/bin/tunnel"   "$@" }
register-old() { "$HOME/Desktop/Create account deepseek/bin/register" "$@" }
```

`register-old` is the previous sign-up automation, a separate project that
launches its own Chrome; leave that line out if you do not have it.

If the project lives somewhere else, change `DEEPSEEK_API_DIR` — that is the
only path the shell config knows. The scripts can also be run directly, e.g.
`./bin/start`, without any shell setup.

---

## Start the server

```bash
start
```

Serves on http://localhost:8010 (and on the LAN, since `HOST=0.0.0.0` in
`.env`). Override the port for one run with `PORT=8080 start`.

Equivalent long form:

```bash
cd ~/Desktop/Deepseek-API && source venv/bin/activate && python app.py
```

## Log in

Opens a browser window to sign in to DeepSeek by hand and saves the session
that the server uses. Run it when there is no saved session yet (fresh
checkout, or after `rm -rf session`) or to switch to a different account.
**Discards the saved session first**, so the login window always appears.
Losing it costs only one sign-in, but any running server will need the new
session, so restart it afterwards.

```bash
login
```

Equivalent long form:

```bash
cd ~/Desktop/Deepseek-API && rm -rf session && source venv/bin/activate && python -m deepseek.auth
```

## Register a fresh account

Creates a brand-new DeepSeek account and saves its session into this project,
inside the Chrome you already have open: the [extension](extension/README.md)
in [extension/](extension/) drives the sign-up page, so no browser is launched
by a script and there is nothing to fingerprint as automation. The Gmail
reading (verification code, next free `+N` alias) happens on the server side.
One-time setup: load `extension/` at `chrome://extensions` (Load unpacked).
The sign-up runs in your normal profile after clearing DeepSeek's site data
there (you get signed out of chat.deepseek.com in that profile); the
credentials stay in the sign-up project's `.env`
(`DEEPSEEK_EMAIL`, `DEEPSEEK_PASSWORD`, `GMAIL_APP_PASSWORD`).

```bash
register
```

Queues the job, prints each step as the extension reports it, and exits 0
once the new session is saved. If the server is running it switches to the
new account on the spot, no restart needed; if it is not, the script hosts
the `/register` routes itself on the same port for the one job and the server
picks up the saved session when it starts. If the extension has not polled
within half a minute, Chrome is started. Any program can trigger the same
thing with `POST http://127.0.0.1:8010/register` on a running server (only
from this machine, not through the tunnel).

`register-old` is the previous way: the sign-up automation in
`~/Desktop/Create account deepseek`, which launches its own Chrome with a
fresh profile and drives it over DevTools (`REGISTER_BROWSER=uc register-old`
for the older SeleniumBase browser). The server's restart fallback still uses
it.

## Muted-account auto-recovery

When DeepSeek answers a request with **"user is muted"**, the account is dead
for good; the only fix is a fresh one, and the server gets one by itself.

**Through the Chrome extension (default).** The server stays up: it queues a
sign-up job for the [extension](extension/README.md), waits one poll interval
for the extension to call, starts Chrome if it does not (so a closed Chrome
is opened, and the extension in it wakes up and polls), then switches to the
new account the moment the session arrives. Nothing restarts and no terminal
is involved; a bare `python app.py` recovers just like `start`. Which Chrome
opens follows `CHROME_BINARY` and `CHROME_PROFILE_DIRECTORY` in `.env`
(defaults: the installed Google Chrome, its last-used profile).

**By restarting (fallback, or `MUTED_RECOVERY=restart`).** The server drops
`logs/.account-muted` and exits; `start` (a supervisor loop) sees the flag,
runs the previous sign-up automation (`register-old`,
`~/Desktop/Create account deepseek/bin/register`) to create a fresh account,
and starts the server again on the new session. The extension path falls
back to this when its job fails (Chrome could not start, no code arrived,
DeepSeek kept refusing after ten attempts).

Safeguards: recoveries are at least 10 minutes apart (in the server for the
extension path, via `logs/.recover-stamp` for `start`), the flag fires at most
once per server lifetime, and register is retried up to 3 times, so a freshly
muted replacement account can't spiral into endless registrations. Ctrl-C and
ordinary crashes exit as before — only a mute triggers any of this.

## HTTPS on your own domain

Serves the API as `https://mody.ddns.net:8010/` with a Let's Encrypt
certificate, on the same port as now.

Let's Encrypt checks that you own the name by fetching a file from
`http://mody.ddns.net/` on port 80; a free No-IP name cannot use the other
check (a DNS TXT record). Port 80 belongs to another PC on the network, so
that PC passes on just the check's path and keeps everything else:

1. On the PC that answers on port 80, in its nginx, inside the server block
   that listens on port 80, add the lines below and reload nginx there. Only
   `/.well-known/acme-challenge/` changes; renewals use it too, so leave it.

   ```nginx
   location ^~ /.well-known/acme-challenge/ {
       proxy_pass http://192.168.1.69:8081;
   }
   ```

2. Run the setup script here (no sudo). It checks the path reaches this
   machine and stops with instructions if not, then gets the certificate,
   writes `SSL_CERTFILE`/`SSL_KEYFILE` into `.env`, and adds a crontab line
   that renews it.

   ```bash
   ./setup-https.sh
   ```

3. Restart the server (Ctrl-C, then `start`).

After that port 8010 serves HTTPS to everyone outside and keeps plain HTTP on
`127.0.0.1:8010`, so the extension, `register` and `tunnel` need no change.
Plain `http://` from outside no longer works. Renewed certificates are
picked up without a restart. `./setup-https.sh --check` only tests step 1.

## Public HTTPS tunnel

Run in a second terminal, with the server already running — it warns you if
nothing is listening. Prints an `https://<random>.trycloudflare.com` URL; the
API base is that URL + `/v1`. The URL changes every restart.

```bash
tunnel
```

Point it elsewhere with `PORT=8080 tunnel`, or at another cloudflared binary
with `CLOUDFLARED=/path/to/cloudflared tunnel`.

Equivalent long form:

```bash
~/cloudflared tunnel --url http://localhost:8010
```
