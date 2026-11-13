// M12 E2E step 4 (video tab): analyse the 30 s clip, open legal lookup for the first recognised sign.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const E2E = resolve('output/demo_20261114/M12/e2e_final');
const SHOTS = resolve(E2E, 'shots');
const PORT = 9333;
const FRONT = 'http://127.0.0.1:8000/';
const VIDEO = resolve(E2E, 'live_30s_source.mp4');
const PROFILE = resolve(E2E, 'chrome-profile');
mkdirSync(SHOTS, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const report = {};

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
let id = 0; const pending = new Map();
ws.addEventListener('message', (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) { const { res, rej } = pending.get(msg.id); pending.delete(msg.id); msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result); }
});
const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expression) => (await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })).result?.value;
const shot = async (name) => { const { data } = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(resolve(SHOTS, name), Buffer.from(data, 'base64')); };
const waitFor = async (expr, ms) => { const end = Date.now() + ms; while (Date.now() < end) { if (await evaluate(expr)) return true; await sleep(1000); } return false; };

await send('Page.enable'); await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: FRONT });
await waitFor("document.querySelector('input[type=file]') !== null", 60000);
await sleep(1000);
const { root } = await send('DOM.getDocument', { depth: -1 });
const { nodeId } = await send('DOM.querySelector', { nodeId: root.nodeId, selector: 'input[type=file]' });
await send('DOM.setFileInputFiles', { nodeId, files: [VIDEO] });
await sleep(800);
await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Phân tích video')?.click()`);
await waitFor(`!!document.querySelector('video.result-video')`, 900000);
await sleep(2000);
report.sign_rows = await evaluate(`[...document.querySelectorAll('.sign-row')].map(r=>r.innerText.replace(/\\s+/g,' ').trim()).slice(0,8)`);
report.chips = await evaluate(`document.querySelectorAll('.sign-row .chip-button').length`);
const clicked = await evaluate(`(()=>{const b=[...document.querySelectorAll('.sign-row .chip-button')][0];if(!b)return false;b.click();return b.innerText})()`);
report.clicked_chip = clicked;
const ok = await waitFor(`!!document.querySelector('.sign-row .excerpt') || [...document.querySelectorAll('.sign-row')].some(r=>r.innerText.includes('Không có đoạn'))`, 60000);
report.lookup_result_visible = ok;
report.lookup_text = await evaluate(`[...document.querySelectorAll('.sign-row')].map(r=>r.innerText.replace(/\\s+/g,' ').trim()).slice(0,3)`);
await shot('10_video_sign_legal_lookup.png');
writeFileSync(resolve(E2E, 'e2e_legal_video_report.json'), JSON.stringify(report, null, 1), 'utf8');
console.log('LEGAL', JSON.stringify(report).slice(0, 600));
ws.close();
chrome.kill();
process.exit(0);
