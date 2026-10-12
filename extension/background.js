// DeepSeek Register — the extension's service worker.
//
// Mirrors ~/Desktop/Create account deepseek/register.py, but inside the Chrome
// you already run: no DevTools port, no chromedriver, nothing launched by a
// script. The local Deepseek-API server (server/register.py) is the broker:
// this worker long-polls it for a job, walks the sign-up form, asks it for
// the verification code (the server reads Gmail), then posts the captured
// session (token + cookies + user agent) back. `python -m server.register`,
// POST /register, the popup's button, or the server itself when DeepSeek
// mutes the account queue the job; the server starts Chrome if this worker
// is not polling. All this does is register and hand over what the API
// needs to use the account.

const DEFAULTS = {
  server: 'http://127.0.0.1:8010',
  token: '',
  trustedInput: false,
  incognito: false,
  signinUrl: 'https://chat.deepseek.com/sign_in',
  signupUrl: 'https://chat.deepseek.com/sign_up',
  chatUrl: 'https://chat.deepseek.com/',
};
const POLL_ALARM = 'deepseek-register-poll';
// Toasts that mean "not this time": the page is refreshed, the form filled
// again and "Send code" clicked again, on the same page.
const RETRY_TEXTS = [
  'Current device environment error',
  'Registration failed. Please try again later',
];
const MAX_ATTEMPTS = 10;
const SCOPE = 'deepseek-register';

const EMAIL_FIELD = 'input[placeholder="Email address"]';
const PASSWORD_FIELD = 'input[placeholder="Password"]';
const CONFIRM_FIELD = 'input[placeholder="Confirm password"]';
const CODE_FIELD = 'input[placeholder="Code"]';
const SEND_CODE = 'span.ds-button__content:contains("Send code")';
const SIGNUP_LINKS = [
  'span.ds-button__content:contains("Sign up")',
  'div[role="button"]:contains("Sign up")',
  'a:contains("Sign up")',
];
const SIGNUP_BUTTONS = [
  'span.ds-button__content:contains("Sign up")',
  'span.ds-button__content:contains("注册")',
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const between = (a, b) => a + Math.random() * (b - a);

let running = null; // the job in progress, if any

// ---- state shown by the popup ----------------------------------------------

async function setState(patch) {
  const { state = {} } = await chrome.storage.session.get('state');
  const next = { ...state, ...patch };
  if (patch.logLine) {
    next.log = [...(state.log || []).slice(-60), `${new Date().toLocaleTimeString()}  ${patch.logLine}`];
    delete next.logLine;
  }
  await chrome.storage.session.set({ state: next });
}

async function settings() {
  const stored = await chrome.storage.local.get(Object.keys(DEFAULTS));
  const s = { ...DEFAULTS };
  for (const k of Object.keys(DEFAULTS)) if (stored[k] !== undefined && stored[k] !== '') s[k] = stored[k];
  s.server = String(s.server).replace(/\/+$/, '');
  return s;
}

// ---- the server ------------------------------------------------------------

class Api {
  constructor(s, jobId) {
    this.base = s.server;
    this.jobId = jobId;
    this.headers = { 'Content-Type': 'application/json' };
    if (s.token) this.headers['X-Register-Token'] = s.token;
  }

  async call(method, path, body, timeoutMs = 28000) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    let res;
    try {
      res = await fetch(this.base + path, {
        method, headers: this.headers, signal: ctl.signal,
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } finally {
      clearTimeout(timer);
    }
    if (res.status === 204) return null;
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { data = { error: text }; }
    if (!res.ok) throw new Error(`${method} ${path} -> ${res.status}: ${(data && data.error) || text}`);
    return data;
  }

  // The alarm fires every 30 s and each poll waits up to 25 s, so the worker
  // is listening most of the time and a job is picked up within seconds.
  static next(s) { return new Api(s).call('GET', '/register/jobs/next?wait=25'); }
  static create(s, reason) { return new Api(s).call('POST', '/register', { reason }); }
  static status(s) { return new Api(s).call('GET', '/register'); }

  async progress(stage, message) {
    await setState({ stage, logLine: message || stage });
    try {
      await this.call('POST', `/register/jobs/${this.jobId}/progress`, { stage, message });
    } catch (e) {
      await setState({ logLine: `(could not report progress: ${e.message})` });
    }
  }

  async waitCode() {
    for (;;) {
      const r = await this.call('GET', `/register/jobs/${this.jobId}/code?wait=25`);
      if (r && r.code) return r.code;
      if (r && r.error) throw new Error(r.error);
    }
  }

  result(payload) { return this.call('POST', `/register/jobs/${this.jobId}/result`, payload); }
  fail(error) { return this.call('POST', `/register/jobs/${this.jobId}/fail`, { error }); }
}

// ---- the page --------------------------------------------------------------

class Driver {
  constructor(tabId, trusted) {
    this.tabId = tabId;
    this.trusted = trusted;
    this.attached = false;
  }

  async ensureContent() {
    for (let i = 0; i < 40; i++) {
      try {
        await chrome.scripting.executeScript({ target: { tabId: this.tabId }, files: ['content.js'] });
        return;
      } catch (e) {
        await sleep(100); // mid-navigation, or the page is not ready yet
      }
    }
    await chrome.scripting.executeScript({ target: { tabId: this.tabId }, files: ['content.js'] });
  }

  async send(cmd, args = {}) {
    await this.ensureContent();
    const reply = await chrome.tabs.sendMessage(this.tabId, { scope: SCOPE, cmd, ...args });
    if (!reply) throw new Error(`no reply from the page for ${cmd}`);
    if (!reply.ok) throw new Error(reply.error);
    return reply.value;
  }

  async waitLoaded(timeoutMs = 30000) {
    const deadline = Date.now() + timeoutMs;
    await sleep(100);
    while (Date.now() < deadline) {
      try {
        const tab = await chrome.tabs.get(this.tabId);
        if (tab.status === 'complete' && (await this.send('readyState')) === 'complete') return;
      } catch (e) { /* still navigating */ }
      await sleep(100);
    }
  }

  async navigate(url) {
    await chrome.tabs.update(this.tabId, { url });
    await this.waitLoaded();
  }

  async reload() {
    await chrome.tabs.reload(this.tabId);
    await this.waitLoaded();
  }

  href() { return this.send('href'); }
  findText(texts) { return this.send('findText', { texts }); }
  isPresent(selector) { return this.send('present', { selector }); }
  readToken() { return this.send('token'); }
  userAgent() { return this.send('ua'); }
  visibleText() { return this.send('visibleText', { limit: 600 }); }

  // First of `selectors` that is on the page, waiting up to `timeoutMs`.
  async waitPresent(selectors, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      for (const sel of selectors) {
        try { if (await this.isPresent(sel)) return sel; } catch (e) { /* navigating */ }
      }
      if (Date.now() >= deadline) return null;
      await sleep(150);
    }
  }

  async click(selector) {
    if (!this.trusted) return this.send('click', { selector });
    const b = await this.send('locate', { selector });
    await sleep(60);
    const x = b.x + b.w * between(0.35, 0.65);
    const y = b.y + b.h * between(0.35, 0.65);
    await this.cdp('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
    await sleep(between(30, 80));
    await this.cdp('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
    await sleep(between(30, 70));
    await this.cdp('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 });
    return true;
  }

  async type(selector, text) {
    if (!this.trusted) return this.send('type', { selector, text });
    await this.click(selector);
    await sleep(between(40, 100));
    if (await this.send('value', { selector })) {
      await this.key('a', 'KeyA', 65, 2); // Ctrl+A
      await this.key('Backspace', 'Backspace', 8, 0);
    }
    for (const ch of text) {
      await this.cdp('Input.dispatchKeyEvent', { type: 'keyDown', key: ch, text: ch, unmodifiedText: ch });
      await this.cdp('Input.dispatchKeyEvent', { type: 'keyUp', key: ch });
      await sleep(between(15, 50));
    }
    return true;
  }

  // Trusted input: real key and mouse events through the DevTools protocol,
  // exactly what register.py sends. Needs the optional "debugger" permission
  // and shows Chrome's "started debugging this browser" bar.
  async cdp(method, params) {
    if (!this.attached) {
      await chrome.debugger.attach({ tabId: this.tabId }, '1.3');
      this.attached = true;
    }
    return chrome.debugger.sendCommand({ tabId: this.tabId }, method, params);
  }

  async key(key, code, keyCode, modifiers) {
    const base = { key, code, modifiers, windowsVirtualKeyCode: keyCode };
    await this.cdp('Input.dispatchKeyEvent', { type: 'rawKeyDown', ...base });
    await this.cdp('Input.dispatchKeyEvent', { type: 'keyUp', ...base });
  }

  async close() {
    if (this.attached) {
      try { await chrome.debugger.detach({ tabId: this.tabId }); } catch (e) { /* gone */ }
      this.attached = false;
    }
  }
}

// ---- the sign-up, step by step --------------------------------------------

function originOf(url) { return new URL(url).origin; }
function baseDomain(url) {
  const host = new URL(url).hostname;
  return host.split('.').slice(-2).join('.');
}

async function openWorkWindow(s) {
  // The sign-up runs in the normal profile: the browser DeepSeek already
  // knows, with its history and its device identity. A new account must not
  // inherit the previous one's login, cookies or storage, so DeepSeek's site
  // data in this profile is wiped first (this signs the profile out of
  // chat.deepseek.com; the API uses the new account's session anyway). The
  // popup's "incognito" setting uses a throwaway incognito window instead.
  const incognito = s.incognito && await chrome.extension.isAllowedIncognitoAccess();
  if (!incognito) {
    const origins = [originOf(s.signinUrl)];
    const base = baseDomain(s.signinUrl);
    if (!origins.includes(`https://${base}`)) origins.push(`https://${base}`);
    await chrome.browsingData.remove({ origins }, {
      cookies: true, localStorage: true, indexedDB: true, cacheStorage: true,
      serviceWorkers: true, fileSystems: true, webSQL: true, cache: true,
    });
  }
  const win = await chrome.windows.create({
    url: 'about:blank', incognito, focused: false, width: 1200, height: 900, type: 'normal',
  });
  const tab = win.tabs && win.tabs[0] ? win.tabs[0]
    : (await chrome.tabs.query({ windowId: win.id }))[0];
  return { win, tab, incognito };
}

// Sign-in page, then its "Sign up" link, as a person would; /sign_up directly
// if the link is not there.
async function openSignupPage(driver, api, s) {
  await driver.navigate(s.signinUrl);
  const link = await driver.waitPresent(SIGNUP_LINKS, 8000);
  if (link) {
    try { await driver.click(link); } catch (e) { /* fall through */ }
  }
  if (!(await driver.waitPresent([EMAIL_FIELD], 8000))) {
    await api.progress('navigate', "'Sign up' link not found, opening /sign_up directly");
    await driver.navigate(s.signupUrl);
    if (!(await driver.waitPresent([EMAIL_FIELD], 10000))) {
      throw new Error('the sign-up form did not appear');
    }
  }
}

// Watches the page for up to `ms` after a click: a retry toast, or (when
// `token` is set) the sign-in token that means it worked.
async function outcome(driver, ms, { token = false } = {}) {
  const deadline = Date.now() + ms;
  for (;;) {
    try {
      const err = await driver.findText(RETRY_TEXTS);
      if (err) return { error: err };
      if (token && await driver.readToken()) return { ok: true };
    } catch (e) { /* the page is navigating: no toast to see */ }
    if (Date.now() >= deadline) return { ok: true };
    await sleep(200);
  }
}

async function fillAndSend(driver, api, job) {
  await api.progress('fill', 'filling the sign-up form');
  await driver.type(EMAIL_FIELD, job.email);
  await driver.type(PASSWORD_FIELD, job.password);
  await driver.type(CONFIRM_FIELD, job.password);
  await sleep(200);
  // The server snapshots the inbox at this stage, so only the email this
  // click produces can be the answer (and any earlier code is forgotten).
  await api.progress('send_code', "clicking 'Send code'");
  await driver.click(SEND_CODE);
  return outcome(driver, 3000);
}

async function enterCodeAndSignup(driver, api, code) {
  await driver.type(CODE_FIELD, code);
  await sleep(200);
  const button = await driver.waitPresent(SIGNUP_BUTTONS, 3000);
  if (!button) throw new Error("'Sign up' button not found after entering the code");
  await driver.click(button);
  await api.progress('submit', "clicked 'Sign up'");
  return outcome(driver, 4000, { token: true });
}

async function captureSession(driver, tab, api, s) {
  let href = '';
  try { href = await driver.href(); } catch (e) { /* mid-navigation */ }
  await api.progress('capture', `capturing the session (at ${href || '?'})`);
  if (!href.startsWith(originOf(s.chatUrl))) {
    await driver.navigate(s.chatUrl);
  }

  let token = null;
  const deadline = Date.now() + 20000;
  let lastDiag = 0;
  while (Date.now() < deadline) {
    try { token = await driver.readToken(); } catch (e) { token = null; }
    if (token) break;
    if (Date.now() - lastDiag > 3000) {
      lastDiag = Date.now();
      let keys = [];
      try { keys = await driver.send('storageKeys'); } catch (e) { /* ignore */ }
      await setState({ logLine: `no token yet (localStorage: ${keys.join(',') || 'empty'})` });
    }
    await sleep(300);
  }

  const stores = await chrome.cookies.getAllCookieStores();
  const store = stores.find((st) => st.tabIds.includes(tab.id));
  const all = await chrome.cookies.getAll(store ? { storeId: store.id } : {});
  const base = baseDomain(s.chatUrl);
  const cookies = {};
  for (const c of all) {
    const domain = c.domain.replace(/^\./, '');
    if (domain === base || domain.endsWith(`.${base}`)) cookies[c.name] = c.value;
  }

  const ua = (await driver.userAgent()) || navigator.userAgent;
  if (!token && Object.keys(cookies).length === 0) {
    throw new Error(`nothing captured (no token, no cookies) at ${await driver.href().catch(() => '?')}`);
  }
  return { token: token || '', cookies, user_agent: ua };
}

async function runJob(job, s) {
  const api = new Api(s, job.id);
  await setState({ status: 'running', job: { id: job.id, email: job.email }, stage: 'starting', log: [] });

  // Chrome stops an idle service worker after 30 s; an extension API call
  // now and then keeps this one alive for the minutes a sign-up takes.
  const keepAlive = setInterval(() => chrome.runtime.getPlatformInfo(), 20000);

  const { win, tab, incognito } = await openWorkWindow(s);
  const driver = new Driver(tab.id, s.trustedInput);
  try {
    await api.progress('window', `window opened (${incognito ? 'incognito' : 'site data cleared'}, `
      + `${s.trustedInput ? 'trusted' : 'synthetic'} input)`);
    await openSignupPage(driver, api, s);

    // Each attempt stays on the sign-up page: on a retry toast the page is
    // refreshed, the form filled again and "Send code" clicked again.
    let error = null;
    for (let n = 1; n <= MAX_ATTEMPTS; n++) {
      if (n > 1) {
        await api.progress('retry', `'${error}' — refreshing the page and trying again (${n}/${MAX_ATTEMPTS})`);
        await driver.reload();
        if (!(await driver.waitPresent([EMAIL_FIELD], 10000))) {
          await openSignupPage(driver, api, s);
        }
      }
      let result = await fillAndSend(driver, api, job);
      if (result.error) { error = result.error; continue; }

      await api.progress('waiting_code', 'waiting for the verification code from Gmail');
      const code = await api.waitCode();
      await api.progress('code', 'code received, entering it');
      result = await enterCodeAndSignup(driver, api, code);
      if (result.error) { error = result.error; continue; }
      error = null;
      break;
    }
    if (error) throw new Error(`DeepSeek answered '${error}' on all ${MAX_ATTEMPTS} attempts`);

    const session = await captureSession(driver, tab, api, s);
    await api.result(session);
    await setState({ status: 'done', stage: 'done', logLine: 'session delivered to the server' });
  } catch (e) {
    let where = '';
    try { where = ` (page said: ${(await driver.visibleText()).replace(/\s+/g, ' ').slice(0, 200)})`; } catch (_) { /* gone */ }
    const error = `${e.message || e}${where}`;
    await setState({ status: 'failed', stage: 'failed', logLine: error });
    try { await api.fail(error); } catch (_) { /* server may be gone */ }
    throw e;
  } finally {
    clearInterval(keepAlive);
    await driver.close();
    try { await chrome.windows.remove(win.id); } catch (_) { /* already closed */ }
  }
}

// ---- polling ---------------------------------------------------------------

async function pollOnce() {
  if (running) return;
  const s = await settings();
  let job;
  try {
    job = await Api.next(s);
    await setState({ serverOk: true, serverError: null, lastPoll: Date.now() });
  } catch (e) {
    await setState({ serverOk: false, serverError: e.message, lastPoll: Date.now() });
    return;
  }
  if (!job) return;
  running = job;
  try {
    await runJob(job, s);
  } catch (e) {
    console.error('register job failed', e);
  } finally {
    running = null;
  }
  pollOnce(); // another job may already be waiting
}

function schedulePolling() {
  chrome.alarms.create(POLL_ALARM, { periodInMinutes: 0.5 });
  pollOnce();
}

chrome.runtime.onInstalled.addListener(schedulePolling);
chrome.runtime.onStartup.addListener(schedulePolling);
chrome.alarms.onAlarm.addListener((a) => { if (a.name === POLL_ALARM) pollOnce(); });

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  // Only the extension's own pages (the popup); content scripts send from
  // the site's URL.
  const own = (sender.url || '').startsWith(chrome.runtime.getURL(''));
  if (!msg || msg.scope !== SCOPE || !own) return false;
  (async () => {
    const s = await settings();
    switch (msg.cmd) {
      case 'poll':
        pollOnce();
        return { ok: true };
      case 'register-now': {
        const job = await Api.create(s, 'extension popup');
        pollOnce();
        return { ok: true, job };
      }
      case 'status':
        return { ok: true, status: await Api.status(s), running: !!running };
      default:
        throw new Error(`unknown command ${msg.cmd}`);
    }
  })().then(reply, (e) => reply({ ok: false, error: String(e.message || e) }));
  return true;
});

// The worker may have been restarted by Chrome: make sure polling is on.
schedulePolling();
