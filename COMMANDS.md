# Commands

## Start the server

```bash
source venv/bin/activate && python app.py
```

Serves on http://localhost:8000 (and on the LAN, since `HOST=0.0.0.0` in `.env`).

## Log in with a different account

Wipes the saved session, then opens a browser to sign in again.

```bash
cd ~/Desktop/Deepseek-API && rm -rf session && source venv/bin/activate && python -m deepseek.auth
```

## Public HTTPS tunnel

Run in a second terminal, with the server already running. Prints an
`https://<random>.trycloudflare.com` URL — the API base is that URL + `/v1`.
The URL changes every restart.

```bash
~/cloudflared tunnel --url http://localhost:8000
```
