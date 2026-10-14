import test from 'node:test';
import assert from 'node:assert/strict';
import {analyzeLatest, freshResult} from './videoRuntime.js';

test('slow inference cannot queue old frames or publish after stop', async () => {
  const video = {ended:false, paused:false, seeking:false, readyState:4, currentTime:0};
  let running = true, captures = 0, release, published = 0;
  const task = analyzeLatest({video, active:()=>running, capture:async()=>++captures,
    analyze:()=>new Promise(resolve=>{release=resolve;}), publish:()=>published++, wait:async()=>{}});
  await new Promise(resolve=>setImmediate(resolve));
  video.currentTime = 2;
  assert.equal(captures, 1);
  running = false;
  release({});
  await task;
  assert.equal(published, 0);
});

test('updates have a 100ms budget instead of an extra fixed 300ms sleep', async () => {
  const video = {ended:false, paused:false, seeking:false, readyState:4, currentTime:1};
  let time = 0, delay, published = 0;
  await analyzeLatest({video, active:()=>!video.ended, capture:async()=>null,
    analyze:async()=>{time+=40; return {};}, publish:()=>published++, now:()=>time,
    wait:async ms=>{delay=ms;video.ended=true;}});
  assert.equal(published, 1);
  assert.equal(delay, 60);
});

test('stale and seek-back overlays disappear', () => {
  assert.equal(freshResult(1, 1.1), true);
  assert.equal(freshResult(1, 1.4), true);
  assert.equal(freshResult(1, 2.4), false);
  assert.equal(freshResult(1, .5), false);
});

test('slow responses update status without showing stale boxes', async () => {
  const video={ended:false,paused:false,seeking:false,readyState:4,currentTime:1};
  let seen, published=0;
  await analyzeLatest({video,active:()=>!video.ended,capture:async()=>null,
    analyze:async()=>{video.currentTime=3;return {};}, publish:()=>published++,
    observe:(_result,_latency,fresh)=>{seen=fresh;}, wait:async()=>{video.ended=true;}});
  assert.equal(seen,false);
  assert.equal(published,0);
});


test('bounded delay allows showing a recently completed lane result',async()=>{
  const video={ended:false,paused:false,seeking:false,readyState:4,currentTime:1};
  let published=0,seen;
  await analyzeLatest({video,active:()=>!video.ended,capture:async()=>null,
    analyze:async()=>{video.currentTime=1.65;return {perception:{lane:{polylines:[]}}};},
    publish:()=>published++,observe:(_result,_latency,fresh)=>{seen=fresh;},
    wait:async()=>{video.ended=true;}});
  assert.equal(seen,true);
  assert.equal(published,1);
});
