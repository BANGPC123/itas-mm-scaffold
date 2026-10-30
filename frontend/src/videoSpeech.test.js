import test from 'node:test';
import assert from 'node:assert/strict';
import {createVideoSpeech} from './videoSpeech.js';

const tick=()=>new Promise(r=>setTimeout(r,0));

test('speech deduplicates observations and is invalidated on seek',async()=>{
  let resolve;const played=[];
  const speech=createVideoSpeech({synthesize:()=>new Promise(r=>{resolve=r;}),play:async url=>played.push(url)});
  const pending=speech.update('Biển cấm',1,()=>1);
  speech.reset();resolve('old.mp3');await pending;assert.deepEqual(played,[]);
  const next=speech.update('Biển cấm',2,()=>2);resolve('new.mp3');await next;
  await speech.update('Biển cấm',3,()=>3);assert.deepEqual(played,['new.mp3']);
});

test('with a queue, a cue that arrives mid-speech is played after the current one',async()=>{
  const played=[];const finish=[];
  const speech=createVideoSpeech({queueWait:2,synthesize:async t=>t,play:(url)=>new Promise(r=>{played.push(url);finish.push(r);})});
  let now=1.2;
  const first=speech.update('cue one',1,()=>now);
  await tick();
  now=1.6;speech.update('cue two',1.5,()=>now);
  assert.deepEqual(played,['cue one']);
  now=1.7;finish[0]();await first;await tick();
  assert.deepEqual(played,['cue one','cue two']);
  finish[1]();
});

test('a queued cue older than queueWait is dropped, and seeking clears the queue',async()=>{
  const played=[];const finish=[];
  const speech=createVideoSpeech({queueWait:2,synthesize:async t=>t,play:(url)=>new Promise(r=>{played.push(url);finish.push(r);})});
  let now=1;
  const first=speech.update('cue one',1,()=>now);
  await tick();
  speech.update('late',1.5,()=>now);
  now=4.5;finish[0]();await first;await tick();
  assert.deepEqual(played,['cue one']);
  const second=speech.update('cue three',5,()=>5);
  await tick();
  speech.update('queued before seek',5.2,()=>5.2);
  speech.reset();finish[1]();await second;await tick();
  assert.deepEqual(played,['cue one','cue three']);
});

test('slow TTS does not play a stale warning',async()=>{
  let now=1;const played=[];
  const speech=createVideoSpeech({synthesize:async()=>{now=10;return 'late.mp3';},play:async u=>played.push(u)});
  await speech.update('Biển cấm',1,()=>now);assert.deepEqual(played,[]);
});
