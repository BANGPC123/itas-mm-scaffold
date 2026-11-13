// M14 E2E, image tab only: app started by start.ps1 (port 8000, ITAS_WARMUP=1 set in start.ps1).
// Upload demo_photo.jpg in the Ảnh tab, analyse, look up the first recognised sign. Headless Chrome on port 9333.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const M14 = resolve('output/demo_20261114/M14');
const E2E = resolve(M14, 'e2e_image');
const SHOTS = resolve(E2E, 'shots');
const PORT = 9333;
const FRONT = 'http://127.0.0.1:8000/';
const PHOTO = resolve(M14, 'demo_photo.jpg');
const PROFILE = resolve(E2E, 'chrome-profile');
mkdirSync(SHOTS, { recursive: true });
mkdirSync(PROFILE, { recursive: true });

const net = [];
const consoleLog = [];
const report = { steps: [] };
const log = (step, data = {}) => { report.steps.push({ t: new Date().toISOString(), step, ...data }); console.log(step, JSON.stringify(data).slice(0, 300)); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1440,900', '--no-first-run', '--disable-features=MediaRouter', 'about:blank',
], { stdio: 'ignore' });

let targets;
for (let i = 0; i < 60; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); if (targets.find((t) => t.type === 'page')) break; } catch {}
  await sleep(500);
}
const page = targets.find((t) => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let id = 0; const pending = new Map(); const handlers = [];
ws.addEventListener('message', (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) { const { res, rej } = pending.get(msg.id); pending.delete(msg.id); msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result); }
  else if (msg.method) handlers.forEach((h) => h(msg));
});
const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expression) => (await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })).result?.value;
const shot = async (name) => { const { data } = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(resolve(SHOTS, name), Buffer.from(data, 'base64')); log('screenshot', { name }); };
const waitFor = async (expr, ms, label) => { const end = Date.now() + ms; while (Date.now() < end) { if (await evaluate(expr)) return true; await sleep(1000); } log('timeout', { label }); return false; };

handlers.push((m) => {
  if (m.method === 'Network.requestWillBeSent') net.push({ kind: 'request', t: m.params.timestamp, url: m.params.request.url, method: m.params.request.method, post: m.params.request.postData?.slice(0, 300) });
  if (m.method === 'Network.responseReceived') net.push({ kind: 'response', t: m.params.timestamp, url: m.params.response.url, status: m.params.response.status, mime: m.params.response.mimeType });
  if (m.method === 'Network.loadingFailed') net.push({ kind: 'failed', t: m.params.timestamp, error: m.params.errorText });
  if (m.method === 'Runtime.consoleAPICalled') consoleLog.push({ type: m.params.type, text: m.params.args.map((a) => a.value ?? a.description).join(' ').slice(0, 300) });
  if (m.method === 'Runtime.exceptionThrown') consoleLog.push({ type: 'exception', text: JSON.stringify(m.params.exceptionDetails).slice(0, 300) });
});

await send('Page.enable'); await send('Network.enable'); await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: FRONT });
await waitFor("document.querySelector('input[type=file]') !== null", 60000, 'upload input');
await sleep(1500);
log('nav', { href: await evaluate('location.href') });

// Switch to the Ảnh tab and upload the photo into the image panel's file input (the last input on the page).
await evaluate(`[...document.querySelectorAll('[role=tab]')].find(b=>b.textContent.trim()==='Ảnh')?.click()`);
await sleep(1000);
const { root } = await send('DOM.getDocument', { depth: -1 });
const inputs = await send('DOM.querySelectorAll', { nodeId: root.nodeId, selector: 'input[type=file]' });
const imageInput = inputs.nodeIds[inputs.nodeIds.length - 1];
await send('DOM.setFileInputFiles', { nodeId: imageInput, files: [PHOTO] });
await sleep(800);
await shot('01_image_selected.png');
const t0 = Date.now();
await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Phân tích ảnh')?.click()`);
const got = await waitFor(`document.querySelector('img.result-image') !== null`, 180000, 'image result');
log('image analysed', { got, seconds: Math.round((Date.now() - t0) / 100) / 10 });
await sleep(3000);
await shot('02_image_result.png');

// Visible sign rows only (the hidden video panel is also in the DOM).
const visibleRows = `[...document.querySelectorAll('.sign-row')].filter(r=>!r.closest('[hidden]'))`;
report.image_sign_rows = await evaluate(`${visibleRows}.map(r=>r.innerText.replace(/\\s+/g,' ').trim())`);
report.image_counter = await evaluate(`(()=>{const el=[...document.querySelectorAll('section *')].find(e=>/nhận dạng/.test(e.textContent)&&e.children.length===0&&!e.closest('[hidden]'));return el?el.textContent.trim():null})()`);
log('image rows', { rows: report.image_sign_rows, counter: report.image_counter });

// Legal lookup on the first recognised (non-unknown) row.
const clicked = await evaluate(`(()=>{const rows=${visibleRows};const r=rows.find(x=>x.querySelector('.chip-button'));const b=r?.querySelector('.chip-button');if(!b)return false;b.click();return b.innerText})()`);
log('chip clicked', { clicked });
const ok = await waitFor(`${visibleRows}.some(r=>r.querySelector('.excerpt')) || ${visibleRows}.some(r=>r.innerText.includes('Không có đoạn'))`, 60000, 'legal lookup');
await sleep(1500);
report.lookup_rows = await evaluate(`${visibleRows}.map(r=>r.innerText.replace(/\\s+/g,' ').trim()).slice(0,3)`);
report.lookup_ok = ok;
await shot('03_image_legal_lookup.png');

const speechReq = net.filter((n) => n.kind === 'request' && n.url.includes('/pipeline/speech')).length;
report.speech_requests_by_ui = speechReq;
report.pipeline_responses = net.filter((n) => n.kind === 'response' && n.url.includes('/pipeline/')).map((n) => ({ url: n.url, status: n.status, mime: n.mime }));
report.console = consoleLog;
writeFileSync(resolve(E2E, 'network_log.json'), JSON.stringify(net, null, 1), 'utf8');
writeFileSync(resolve(E2E, 'console_log.json'), JSON.stringify(consoleLog, null, 1), 'utf8');
writeFileSync(resolve(E2E, 'e2e_image_report.json'), JSON.stringify(report, null, 1), 'utf8');
log('done', { lookup_ok: ok, console_entries: consoleLog.length });
ws.close();
chrome.kill();
process.exit(0);
