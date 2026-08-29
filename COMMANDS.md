# Commands

Three shortcuts do the everyday jobs. They work from any directory — each one
finds the project itself — and are defined as zsh functions in `~/.zshrc`
pointing at the scripts in [bin/](bin/):

| Command    | What it does                                        |
| ---------- | --------------------------------------------------- |
| `start`    | Run the API server (foreground; Ctrl-C stops it)    |
| `register` | Sign in to DeepSeek again, as a different account   |
| `tunnel`   | Expose the local server on a public HTTPS URL       |

To set them up on another machine, add this to `~/.zshrc` (or `~/.bashrc`) and
open a new terminal:

```bash
export DEEPSEEK_API_DIR="$HOME/Desktop/Deepseek-API"
start()    { "$DEEPSEEK_API_DIR/bin/start"    "$@" }
register() { "$DEEPSEEK_API_DIR/bin/register" "$@" }
tunnel()   { "$DEEPSEEK_API_DIR/bin/tunnel"   "$@" }
```

If the project lives somewhere else, change `DEEPSEEK_API_DIR` — that is the
only path the shell config knows. The scripts can also be run directly, e.g.
`./bin/start`, without any shell setup.

---

## Start the server

```bash
start
```

Serves on http://localhost:8000 (and on the LAN, since `HOST=0.0.0.0` in
`.env`). Override the port for one run with `PORT=8080 start`.

Equivalent long form:

```bash
cd ~/Desktop/Deepseek-API && source venv/bin/activate && python app.py
```

## Log in with a different account

**Discards the saved session**, then opens a browser to sign in again. Losing
it costs only one sign-in, but any running server will need the new session, so
restart it afterwards.

```bash
register
```

Equivalent long form:

```bash
cd ~/Desktop/Deepseek-API && rm -rf session && source venv/bin/activate && python -m deepseek.auth
```

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
~/cloudflared tunnel --url http://localhost:8000
```
