// M12 final E2E: app started by start.ps1 (port 8000); headless Chrome over DevTools (port 9333).
// Follows the DEMO_SCRIPT steps: video upload + warnings, speech request, image tab, legal lookup.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const E2E = resolve('output/demo_20261114/M12/e2e_final');
const SHOTS = resolve(E2E, 'shots');
const PORT = 9333;
const FRONT = 'http://127.0.0.1:8000/';
const VIDEO = resolve(E2E, 'live_30s_source.mp4');
const PHOTO = resolve(E2E, 'photo_61s.jpg');
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
  if (m.method === 'Network.requestWillBeSent') net.push({ kind: 'request', t: m.params.timestamp, url: m.params.request.url, method: m.params.request.method, post: m.params.request.postData?.slice(0, 300) });
  if (m.method === 'Network.responseReceived') net.push({ kind: 'response', t: m.params.timestamp, url: m.params.response.url, status: m.params.response.status, mime: m.params.response.mimeType });
  if (m.method === 'Network.loadingFailed') net.push({ kind: 'failed', t: m.params.timestamp, error: m.params.errorText });
  if (m.method === 'Runtime.consoleAPICalled') consoleLog.push({ type: m.params.type, text: m.params.args.map((a) => a.value ?? a.description).join(' ').slice(0, 300) });
  if (m.method === 'Runtime.exceptionThrown') consoleLog.push({ type: 'exception', text: JSON.stringify(m.params.exceptionDetails).slice(0, 300) });
  if (m.method === 'Log.entryAdded') consoleLog.push({ type: 'log-' + m.params.entry.level, text: m.params.entry.text.slice(0, 300), url: m.params.entry.url });
});

await send('Page.enable'); await send('Network.enable'); await send('Runtime.enable'); await send('Log.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: FRONT });
await waitFor("document.querySelector('input[type=file]') !== null", 60000, 'upload input');
await sleep(1500);
await shot('01_home_video_tab.png');
log('nav', { href: await evaluate('location.href') });

// DEMO step 2: upload the 30 s live clip in the video tab and analyse it.
const { root } = await send('DOM.getDocument', { depth: -1 });
const { nodeId } = await send('DOM.querySelector', { nodeId: root.nodeId, selector: 'input[type=file]' });
await send('DOM.setFileInputFiles', { nodeId, files: [VIDEO] });
await sleep(1000);
await shot('02_video_selected.png');
await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Phân tích video')?.click()`);
log('analyze clicked');
await waitFor(`document.body.innerText.includes('Đang phân tích video') || document.body.innerText.includes('Đang chờ')`, 60000, 'progress visible');
await sleep(3000);
await shot('03_video_progress.png');
const done = await waitFor(`!!document.querySelector('video.result-video')`, 900000, 'result video');
await sleep(2000);
await shot('04_video_result.png');
log('result shown', { done });

// Forward-warning list: heading, count and rows.
const forward = await evaluate(`(()=>{const h=[...document.querySelectorAll('h2')].find(x=>x.textContent.trim()==='Cảnh báo phía trước');if(!h)return null;const card=h.closest('section');return {heading:true,count:card.querySelector('.count')?.innerText,rows:[...card.querySelectorAll('.sign-row')].map(r=>r.innerText.replace(/\s+/g,' ').trim())}})()`);
log('forward list', forward || { heading: false });
await evaluate(`[...document.querySelectorAll('h2')].find(x=>x.textContent.trim()==='Cảnh báo phía trước')?.scrollIntoView()`);
await sleep(800);
await shot('05_forward_list.png');

// DEMO step 2 (playback): play the result from the start; cues run through the UI.
await evaluate(`document.querySelector('video.result-video').currentTime=0; document.querySelector('video.result-video').play().catch(()=>{})`);
await sleep(12000);
await shot('06_video_playing_12s.png');
await sleep(20000);
await shot('07_video_playing_32s.png');
const videoState = await evaluate(`(()=>{const v=document.querySelector('video.result-video');return {t:v.currentTime,paused:v.paused,ended:v.ended}})()`);
log('video state', videoState);

// Explicit speech check for the sentence "Chú ý xe máy" (same API the UI calls).
const speech = await evaluate(`(async()=>{const r=await fetch('/pipeline/speech',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:'Chú ý xe máy'})});const text=await r.text();let j=null;try{j=JSON.parse(text)}catch{};if(!j)return {speech_status:r.status,body:text.slice(0,200)};const a=await fetch(j.audio_url);const b=await a.arrayBuffer();return {speech_status:r.status,audio_url:j.audio_url,audio_status:a.status,audio_bytes:b.byteLength,content_type:a.headers.get('content-type')}})()`);
log('speech check', speech);

// DEMO step 3: image tab.
await evaluate(`[...document.querySelectorAll('[role=tab]')].find(b=>b.textContent.trim()==='Ảnh')?.click()`);
await sleep(1000);
const inputs = await send('DOM.querySelectorAll', { nodeId: root.nodeId, selector: 'input[type=file]' });
const imageInput = inputs.nodeIds[inputs.nodeIds.length - 1];
await send('DOM.setFileInputFiles', { nodeId: imageInput, files: [PHOTO] });
await sleep(800);
await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Phân tích ảnh')?.click()`);
await waitFor(`document.querySelector('img.result-image') !== null`, 180000, 'image result');
await sleep(4000);
await shot('08_image_result.png');

// DEMO step 4: legal lookup on the first sign row.
await evaluate(`(()=>{const b=[...document.querySelectorAll('.sign-row .chip-button')][0];b?.click();return !!b})()`);
await waitFor(`!!document.querySelector('.sign-row .excerpt') || !!document.querySelector('[role=alert]') || [...document.querySelectorAll('.sign-row')].some(r=>r.innerText.includes('Không có đoạn'))`, 60000, 'legal lookup');
await shot('09_legal_lookup.png');
await sleep(1000);

const speechReq = net.filter((n) => n.kind === 'request' && n.url.includes('/pipeline/speech')).map((n) => { try { return JSON.parse(n.post).text; } catch { return n.post; } });
const pipelineResp = net.filter((n) => n.kind === 'response' && n.url.includes('/pipeline/')).map((n) => ({ url: n.url, status: n.status, mime: n.mime }));
report.speech_texts_requested_by_ui = speechReq;
report.pipeline_responses = pipelineResp;
report.console = consoleLog;
writeFileSync(resolve(E2E, 'network_log.json'), JSON.stringify(net, null, 1), 'utf8');
writeFileSync(resolve(E2E, 'console_log.json'), JSON.stringify(consoleLog, null, 1), 'utf8');
writeFileSync(resolve(E2E, 'e2e_report.json'), JSON.stringify({ ...report, forward, videoState, speech }, null, 1), 'utf8');
log('done', { ui_speech_requests: speechReq.length, console_entries: consoleLog.length });
ws.close();
chrome.kill();
process.exit(0);
