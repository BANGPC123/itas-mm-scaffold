import test from 'node:test';
import assert from 'node:assert/strict';
import {createVideoSpeech} from './videoSpeech.js';

test('speech deduplicates observations and is invalidated on seek',async()=>{
  let resolve;const played=[];
  const speech=createVideoSpeech({synthesize:()=>new Promise(r=>{resolve=r;}),play:async url=>played.push(url)});
  const pending=speech.update('Biển cấm',1,()=>1);
  speech.reset();resolve('old.mp3');await pending;assert.deepEqual(played,[]);
  const next=speech.update('Biển cấm',2,()=>2);resolve('new.mp3');await next;
  await speech.update('Biển cấm',3,()=>3);assert.deepEqual(played,['new.mp3']);
});

test('slow TTS does not play a stale warning',async()=>{
  let now=1;const played=[];
  const speech=createVideoSpeech({synthesize:async()=>{now=10;return 'late.mp3';},play:async u=>played.push(u)});
  await speech.update('Biển cấm',1,()=>now);assert.deepEqual(played,[]);
});
