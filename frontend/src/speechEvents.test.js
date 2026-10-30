import test from 'node:test';
import assert from 'node:assert/strict';
import {takeDueEvents,handledBefore} from './speechEvents.js';

const events=[{t_seconds:1,text:'a'},{t_seconds:4,text:'b'},{t_seconds:9,text:'c'}];

test('events are taken once, in time order, when playback reaches them',()=>{
  const handled=new Set();
  assert.deepEqual(takeDueEvents(events,0.5,handled),[]);
  assert.deepEqual(takeDueEvents(events,4.2,handled).map(e=>e.text),['a','b']);
  assert.deepEqual(takeDueEvents(events,4.3,handled),[]);
  assert.deepEqual(takeDueEvents(events,10,handled).map(e=>e.text),['c']);
});

test('seeking forward skips passed cues; seeking back keeps later cues pending',()=>{
  const forward=handledBefore(events,5);
  assert.deepEqual(takeDueEvents(events,5.1,forward).map(e=>e.text),[]);
  const back=handledBefore(events,2);
  assert.deepEqual(takeDueEvents(events,4.5,back).map(e=>e.text),['b']);
});
