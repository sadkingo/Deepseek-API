// Runs inside the DeepSeek page and does what register.py's RealChrome does
// from outside: find elements (CSS, plus SeleniumBase's :contains("text")),
// click them, type into them, and read the page. Injected by background.js
// before every command, so it must be safe to inject twice.
(() => {
  if (window.__deepseekRegister) return;
  window.__deepseekRegister = true;

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const between = (a, b) => a + Math.random() * (b - a);

  // First element matching `sel`: plain CSS, or CSS followed by
  // :contains("text") for text anywhere inside the element.
  function find(sel, visibleOnly) {
    const m = sel.match(/^(.*?):contains\((['"])([\s\S]*)\2\)\s*$/);
    const css = m ? (m[1] || '*') : sel;
    const text = m ? m[3] : null;
    let els = Array.from(document.querySelectorAll(css));
    if (text !== null) els = els.filter((e) => (e.textContent || '').includes(text));
    if (visibleOnly) {
      els = els.filter((e) => {
        const r = e.getBoundingClientRect();
        const s = getComputedStyle(e);
        return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
      });
    }
    return els.length ? els[0] : null;
  }

  async function waitFor(sel, visibleOnly, timeout) {
    const deadline = Date.now() + (timeout ?? 10000);
    for (;;) {
      const el = find(sel, visibleOnly);
      if (el) return el;
      if (Date.now() >= deadline) throw new Error(`element not found: ${sel}`);
      await sleep(150);
    }
  }

  function box(el) {
    el.scrollIntoView({ block: 'center', inline: 'center' });
    const r = el.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height };
  }

  // React only notices a value set through the element's own prototype
  // setter, followed by an input event.
  function setNativeValue(el, value) {
    const proto = el instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, value);
  }

  function mouseOpts(x, y, extra) {
    return Object.assign({
      bubbles: true, cancelable: true, composed: true, view: window,
      clientX: x, clientY: y, screenX: x, screenY: y, button: 0,
    }, extra || {});
  }

  async function click(el) {
    const b = box(el);
    await sleep(60);
    const x = b.x + b.w * between(0.35, 0.65);
    const y = b.y + b.h * between(0.35, 0.65);
    const target = document.elementFromPoint(x, y) || el;
    const ptr = { pointerId: 1, pointerType: 'mouse', isPrimary: true };
    target.dispatchEvent(new PointerEvent('pointermove', mouseOpts(x, y, ptr)));
    target.dispatchEvent(new MouseEvent('mousemove', mouseOpts(x, y)));
    await sleep(between(30, 80));
    target.dispatchEvent(new PointerEvent('pointerdown', mouseOpts(x, y, { ...ptr, buttons: 1 })));
    target.dispatchEvent(new MouseEvent('mousedown', mouseOpts(x, y, { buttons: 1 })));
    if (typeof el.focus === 'function') el.focus();
    await sleep(between(30, 70));
    target.dispatchEvent(new PointerEvent('pointerup', mouseOpts(x, y, ptr)));
    target.dispatchEvent(new MouseEvent('mouseup', mouseOpts(x, y)));
    target.dispatchEvent(new MouseEvent('click', mouseOpts(x, y, { detail: 1 })));
  }

  function keyOpts(ch) {
    const code = /^[a-z]$/i.test(ch) ? `Key${ch.toUpperCase()}`
      : /^[0-9]$/.test(ch) ? `Digit${ch}` : ch === ' ' ? 'Space' : '';
    return { key: ch, code, bubbles: true, cancelable: true, composed: true };
  }

  async function type(el, text) {
    await click(el);
    await sleep(between(40, 100));
    el.focus();
    if (el.value) {
      setNativeValue(el, '');
      el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' }));
    }
    let value = '';
    for (const ch of text) {
      el.dispatchEvent(new KeyboardEvent('keydown', keyOpts(ch)));
      el.dispatchEvent(new KeyboardEvent('keypress', keyOpts(ch)));
      value += ch;
      setNativeValue(el, value);
      el.dispatchEvent(new InputEvent('input', {
        bubbles: true, composed: true, data: ch, inputType: 'insertText',
      }));
      el.dispatchEvent(new KeyboardEvent('keyup', keyOpts(ch)));
      await sleep(between(15, 50));
    }
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  async function handle(msg) {
    switch (msg.cmd) {
      case 'ping': return true;
      case 'href': return location.href;
      case 'ua': return navigator.userAgent;
      case 'readyState': return document.readyState;
      // Text a person could see, not the page's markup or scripts.
      case 'hasText': return ((document.body && document.body.innerText) || '').includes(msg.text);
      case 'findText': {
        const shown = (document.body && document.body.innerText) || '';
        return msg.texts.find((t) => shown.includes(t)) || null;
      }
      case 'visibleText': return (document.body && document.body.innerText || '').slice(0, msg.limit || 2000);
      case 'present': return find(msg.selector, false) !== null;
      case 'locate': return box(await waitFor(msg.selector, true, msg.timeout));
      case 'click': await click(await waitFor(msg.selector, true, msg.timeout)); return true;
      case 'type': await type(await waitFor(msg.selector, true, msg.timeout), msg.text); return true;
      case 'focus': (await waitFor(msg.selector, true, msg.timeout)).focus(); return true;
      case 'value': return (await waitFor(msg.selector, false, msg.timeout)).value;
      case 'token': {
        try {
          const raw = localStorage.getItem('userToken');
          return raw ? JSON.parse(raw).value : null;
        } catch (e) { return null; }
      }
      case 'storageKeys': {
        try { return Object.keys(localStorage); } catch (e) { return [`ERR:${e.message}`]; }
      }
      default: throw new Error(`unknown command ${msg.cmd}`);
    }
  }

  chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
    if (!msg || msg.scope !== 'deepseek-register') return false;
    handle(msg).then(
      (value) => reply({ ok: true, value }),
      (e) => reply({ ok: false, error: String((e && e.message) || e) }),
    );
    return true;
  });
})();
