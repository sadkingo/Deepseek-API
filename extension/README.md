# DeepSeek Register — Chrome extension

Creates a fresh DeepSeek account inside the Chrome you already use, on request
from the local Deepseek-API server, and hands the server the new session. It
does what `register` (the sign-up automation in
`~/Desktop/Create account deepseek`) does, with one difference: nothing
launches Chrome. There is no DevTools port, no chromedriver and no automation
switch, because the browser is the one you opened yourself.

## How it works

```
bin/register ──POST /register──▶ server ◀──long-poll── extension (Chrome)
                                            │  ▲                    │
                     Gmail (IMAP) ◀─────────┘  └── token+cookies ◀──┘ sign-up form
```

1. Something queues a job: `bin/register`,
   `POST http://127.0.0.1:8010/register`, the popup's **Register now** button,
   or the server itself the moment DeepSeek mutes the account. If the
   extension has not polled within a poll interval, the server (or the
   script) starts Chrome, so a closed Chrome is no obstacle.
2. The server resolves the next free `+N` Gmail alias (as `register.py` does)
   and holds the job. The extension polls every 30 s and picks it up.
3. The extension clears DeepSeek's site data in your profile (so the old
   login cannot get in the way), opens a normal window, goes to the sign-in
   page, follows
   **Sign up**, types the email and password, and clicks **Send code**. On
   "Current device environment error" or "Registration failed. Please try
   again later" it refreshes that page, fills the form again and clicks
   **Send code** again, up to ten times. The same goes for a toast after the
   final **Sign up**.
4. It asks the server for the code; the server reads it from Gmail over IMAP
   and answers as soon as the email arrives.
5. It types the code, clicks **Sign up**, waits for the token to appear in
   `localStorage`, collects the `deepseek.com` cookies and the user agent, and
   posts them. The server writes `session/session.json`, points the sign-up
   project's `.env` at the new address, and switches to the new account
   without a restart.

## Install (once)

1. Open `chrome://extensions`, turn on **Developer mode**, click **Load
   unpacked** and pick this `extension/` folder.
2. Click the extension's icon. The server URL defaults to
   `http://127.0.0.1:8010`; change it if the server runs on another port, fill
   in the token if `REGISTER_TOKEN` is set on the server, and **Save**.

Chrome keeps the extension's service worker asleep between polls; it wakes
every 30 s while Chrome is open. Chrome must be open for a job to run.

## Trigger it

From the terminal:

```bash
bin/register
```

It queues the job, prints each step as the extension reports it, and exits 0
once the session is saved. The API server does not have to be running: the
script then hosts the `/register` routes itself on the same port for that
one job, and the server loads the saved session when it starts. Or, with the
server running, from anything that can make an HTTP request:

```bash
curl -X POST http://127.0.0.1:8010/register
curl http://127.0.0.1:8010/register/jobs/<id>      # state, stage, log
curl -X POST http://127.0.0.1:8010/register -d '{"wait": 600}'   # block until done
```

The routes answer only direct requests from this machine: not through the
Cloudflare tunnel or another proxy, not from a LAN client, and not from a web
page (the job carries the sign-up password). `REGISTER_TOKEN=<secret>` in the
server's `.env` additionally requires an `X-Register-Token` header, which the
popup lets you set.

## When DeepSeek mutes the account

Nothing to do. The server queues a job for the extension, starts Chrome if
the extension is not polling (`CHROME_BINARY` and `CHROME_PROFILE_DIRECTORY`
in the server's `.env` choose which Chrome and profile; the extension must be
installed in that profile), and switches to the new account when the session
arrives, all without going down. If that fails it falls back to the old
exit-and-restart recovery in `bin/start`. Details in
[COMMANDS.md](../COMMANDS.md#muted-account-auto-recovery).

## Settings

| Setting | Meaning |
| --- | --- |
| Server | The Deepseek-API server to poll (default `http://127.0.0.1:8010`). |
| Register token | Sent as `X-Register-Token`; only needed with `REGISTER_TOKEN`. |
| Use an incognito window | Off by default: the sign-up runs in your normal profile, the browser DeepSeek already knows, after clearing DeepSeek's site data there (you get signed out of chat.deepseek.com in that profile; the API uses the new account anyway). On: a throwaway incognito window, which needs **Allow in Incognito** on the extension's page. |
| Trusted input | Type and click through the DevTools protocol (`chrome.debugger`), so the page sees real, trusted input events exactly as `register.py` sends them. Asks for the optional `debugger` permission and shows Chrome's "started debugging this browser" bar during a sign-up. Off by default: the page then gets synthetic pointer, keyboard and input events, which React forms accept. Turn it on if DeepSeek keeps answering with the device error. |

`chrome.storage.local` also honours `signinUrl`, `signupUrl` and `chatUrl`,
which the end-to-end test uses to point the extension at a mock site.

## Test

```bash
venv/bin/python -m unittest tests.test_register                        # the server side
RUN_EXTENSION_E2E=1 venv/bin/python -m unittest tests.test_extension_e2e  # the extension in Chromium, ~30 s
```

The end-to-end test loads this folder into Playwright's Chromium, serves a
mock DeepSeek whose sign-up page has the same fields, fakes Gmail, and checks
the whole exchange, including a device-error retry.
