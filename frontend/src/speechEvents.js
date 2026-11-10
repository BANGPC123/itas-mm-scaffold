// Cue events from an offline video report: {t_seconds, codes, text}.
// `handled` is a Set of event objects already passed to the speaker.
export function takeDueEvents(events,currentTime,handled){
  const due=[];
  for(const event of events){
    if(event.t_seconds<=currentTime&&!handled.has(event)){handled.add(event);due.push(event);}
  }
  return due;
}

// After a seek, events before the new position count as passed; later ones are still pending.
export function handledBefore(events,time){
  return new Set(events.filter(event=>event.t_seconds<time));
}
