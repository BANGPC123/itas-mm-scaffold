// M16 E2E on main (app started by start.ps1, port 8000). Headless Chrome on port 9333.
// Video tab: live 30 s source -> 2 forward warnings, 1 spoken; speech request for "Chú ý xe máy".
// Image tab: M14 demo photo -> I.423 recognised; legal lookup on the visible row.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve('.');
const E2E = resolve(ROOT, 'output/demo_20261114/M16/e2e_main');
const SHOTS = resolve(E2E, 'shots');
const PORT = 9333;
const FRONT = 'http://127.0.0.1:8000/';
const VIDEO = resolve(ROOT, 'output/demo_20261114/M12/e2e_final/live_30s_source.mp4');
const PHOTO = resolve(ROOT, 'output/demo_20261114/M14/demo_photo.jpg');
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
  '--window-size=1440,900', '--autoplay-policy=no-user-gesture-required', '--no-first-run',
  '--disable-features=MediaRouter', 'about:blank',
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
  if (m.method === 'Network.requestWillBeSent') net.push({ kind: 'request', url: m.params.request.url, method: m.params.request.method, post: m.params.request.postData?.slice(0, 300) });
  if (m.method === 'Network.responseReceived') net.push({ kind: 'response', url: m.params.response.url, status: m.params.response.status, mime: m.params.response.mimeType });
  if (m.method === 'Network.loadingFailed') net.push({ kind: 'failed', error: m.params.errorText });
  if (m.method === 'Runtime.consoleAPICalled') consoleLog.push({ type: m.params.type, text: m.params.args.map((a) => a.value ?? a.description).join(' ').slice(0, 300) });
  if (m.method === 'Runtime.exceptionThrown') consoleLog.push({ type: 'exception', text: JSON.stringify(m.params.exceptionDetails).slice(0, 300) });
});

await send('Page.enable'); await send('Network.enable'); await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: FRONT });
await waitFor("document.querySelector('input[type=file]') !== null", 60000, 'upload input');
await sleep(1500);
await shot('01_home.png');

// ---- Video tab
const { root } = await send('DOM.getDocument', { depth: -1 });
const { nodeId } = await send('DOM.querySelector', { nodeId: root.nodeId, selector: 'input[type=file]' });
await send('DOM.setFileInputFiles', { nodeId, files: [VIDEO] });
await sleep(1000);
await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Phân tích video')?.click()`);
const vDone = await waitFor(`!!document.querySelector('video.result-video')`, 900000, 'video result');
await sleep(2000);
await shot('02_video_result.png');
const forward = await evaluate(`(()=>{const h=[...document.querySelectorAll('h2')].find(x=>x.textContent.trim()==='Cảnh báo phía trước');if(!h)return null;const card=h.closest('section');return {count:card.querySelector('.count')?.innerText,rows:[...card.querySelectorAll('.sign-row')].map(r=>r.innerText.replace(/\\s+/g,' ').trim())}})()`);
log('video forward list', { vDone, forward });
await evaluate(`[...document.querySelectorAll('h2')].find(x=>x.textContent.trim()==='Cảnh báo phía trước')?.scrollIntoView()`);
await sleep(800);
await shot('03_forward_list.png');
const speech = await evaluate(`(async()=>{const r=await fetch('/pipeline/speech',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:'Chú ý xe máy'})});const j=await r.json();const a=await fetch(j.audio_url);const b=await a.arrayBuffer();return {speech_status:r.status,audio_status:a.status,audio_bytes:b.byteLength,content_type:a.headers.get('content-type')}})()`);
log('speech check', speech);
report.video = { vDone, forward, speech };

// ---- Image tab
await evaluate(`[...document.querySelectorAll('[role=tab]')].find(b=>b.textContent.trim()==='Ảnh')?.click()`);
await sleep(1000);
const inputs = await send('DOM.querySelectorAll', { nodeId: root.nodeId, selector: 'input[type=file]' });
const imageInput = inputs.nodeIds[inputs.nodeIds.length - 1];
await send('DOM.setFileInputFiles', { nodeId: imageInput, files: [PHOTO] });
await sleep(800);
await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Phân tích ảnh')?.click()`);
const iDone = await waitFor(`document.querySelector('img.result-image') !== null`, 180000, 'image result');
await sleep(3000);
await shot('04_image_result.png');
const visibleRows = `[...document.querySelectorAll('.sign-row')].filter(r=>!r.closest('[hidden]'))`;
const imgRows = await evaluate(`${visibleRows}.map(r=>r.innerText.replace(/\\s+/g,' ').trim())`);
const imgCounter = await evaluate(`(()=>{const el=[...document.querySelectorAll('section *')].find(e=>/nhận dạng/.test(e.textContent)&&e.children.length===0&&!e.closest('[hidden]'));return el?el.textContent.trim():null})()`);
log('image rows', { iDone, imgRows, imgCounter });
const clicked = await evaluate(`(()=>{const rows=${visibleRows};const r=rows.find(x=>x.querySelector('.chip-button'));const b=r?.querySelector('.chip-button');if(!b)return false;b.click();return b.innerText})()`);
const lookupOk = await waitFor(`${visibleRows}.some(r=>r.querySelector('.excerpt')) || ${visibleRows}.some(r=>r.innerText.includes('Không có đoạn'))`, 60000, 'legal lookup');
await sleep(1500);
const lookup = await evaluate(`${visibleRows}.map(r=>r.innerText.replace(/\\s+/g,' ').trim()).slice(0,1)`);
await shot('05_image_legal_lookup.png');
report.image = { iDone, imgRows, imgCounter, clicked, lookupOk, lookup };
log('image lookup', { clicked, lookupOk, head: String(lookup).slice(0, 160) });

report.network = net.filter((n) => n.kind === 'response' && n.url.includes('/pipeline/')).map((n) => ({ url: n.url, status: n.status, mime: n.mime }));
report.console = consoleLog;
writeFileSync(resolve(E2E, 'network_log.json'), JSON.stringify(net, null, 1), 'utf8');
writeFileSync(resolve(E2E, 'console_log.json'), JSON.stringify(consoleLog, null, 1), 'utf8');
writeFileSync(resolve(E2E, 'e2e_main_report.json'), JSON.stringify(report, null, 1), 'utf8');
log('done', { console_entries: consoleLog.length });
ws.close();
chrome.kill();
process.exit(0);
