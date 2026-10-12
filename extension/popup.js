const SCOPE = 'deepseek-register';
const $ = (id) => document.getElementById(id);

function send(cmd) {
  return chrome.runtime.sendMessage({ scope: SCOPE, cmd });
}

async function loadSettings() {
  const s = await chrome.storage.local.get(['server', 'token', 'trustedInput', 'incognito']);
  $('server').value = s.server || 'http://127.0.0.1:8010';
  $('token').value = s.token || '';
  $('trusted').checked = !!s.trustedInput;
  $('incognito').checked = !!s.incognito;
}

async function save() {
  let trusted = $('trusted').checked;
  if (trusted) {
    // Trusted input drives the page through chrome.debugger, an optional
    // permission Chrome only grants on a click like this one.
    trusted = await chrome.permissions.request({ permissions: ['debugger'] });
    $('trusted').checked = trusted;
  }
  await chrome.storage.local.set({
    server: $('server').value.trim().replace(/\/+$/, ''),
    token: $('token').value.trim(),
    trustedInput: trusted,
    incognito: $('incognito').checked,
  });
  await refresh();
}

function fmtAgo(ts) {
  if (!ts) return 'never';
  const s = Math.round((Date.now() - ts) / 1000);
  return s < 60 ? `${s}s ago` : `${Math.round(s / 60)} min ago`;
}

async function refresh() {
  const allowed = await chrome.extension.isAllowedIncognitoAccess();
  if (!$('incognito').checked) {
    $('incognito-line').innerHTML = 'Window: <span class="ok">normal profile</span> — DeepSeek site data '
      + 'in this profile is cleared before each sign-up (you get signed out of chat.deepseek.com).';
  } else if (allowed) {
    $('incognito-line').innerHTML = 'Window: <span class="ok">incognito</span> — each sign-up runs in a fresh incognito window.';
  } else {
    $('incognito-line').innerHTML = 'Window: <span class="warn">incognito not allowed</span> — a normal window '
      + 'will be used. <a id="ext-page">Allow in Incognito</a> on the extension page.';
  }
  const link = $('ext-page');
  if (link) link.onclick = () => chrome.tabs.create({ url: `chrome://extensions/?id=${chrome.runtime.id}` });

  let status = null;
  try {
    const r = await send('status');
    if (!r || !r.ok) throw new Error((r && r.error) || 'no reply');
    status = r.status;
    $('server-line').innerHTML = `<span class="ok">Server reachable</span> (${$('server').value})`;
    $('creds-line').textContent = status.credentials_ok
      ? 'Sign-up credentials: configured'
      : `Sign-up credentials missing: ${status.credentials_missing.join(', ')} in ${status.account_env}`;
    $('creds-line').className = status.credentials_ok ? 'ok' : 'bad';
    $('register').disabled = !status.credentials_ok;
  } catch (e) {
    $('server-line').innerHTML = `<span class="bad">Server unreachable:</span> ${e.message}`;
    $('creds-line').textContent = '';
    $('register').disabled = true;
  }

  const { state = {} } = await chrome.storage.session.get('state');
  const cur = status && status.current;
  const parts = [];
  if (cur) parts.push(`Job ${cur.id}: ${cur.state} / ${cur.stage} (${cur.email})`);
  else parts.push('No job queued.');
  parts.push(`Last poll: ${fmtAgo(state.lastPoll)}.`);
  $('job-line').textContent = parts.join(' ');
  const lines = [...(state.log || [])];
  if (cur && cur.error) lines.push(`error: ${cur.error}`);
  $('log').textContent = lines.join('\n');
}

$('save').onclick = save;
$('poll').onclick = async () => { await send('poll'); setTimeout(refresh, 800); };
$('register').onclick = async () => {
  $('register').disabled = true;
  const r = await send('register-now');
  if (!r || !r.ok) $('job-line').textContent = `Could not queue a job: ${(r && r.error) || 'no reply'}`;
  setTimeout(refresh, 800);
};

loadSettings().then(refresh);
setInterval(refresh, 3000);
