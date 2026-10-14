export function freshResult(sampleTime, currentTime, ageSeconds = 1.2) {
  return currentTime >= sampleTime && currentTime - sampleTime <= ageSeconds;
}

export async function analyzeLatest({video, capture, analyze, publish, active, observe = () => {},
  wait = ms => new Promise(resolve => setTimeout(resolve, ms)), now = () => performance.now(), interval = 100}) {
  while (active() && !video.ended) {
    if (video.paused || video.seeking || video.readyState < 2) {
      await wait(50);
      continue;
    }
    const started = now(), timestamp = video.currentTime;
    const blob = await capture();
    if (!active()) break;
    try {
      const result = await analyze(blob, timestamp);
      if (active()) {
        const fresh = !video.seeking && freshResult(timestamp, video.currentTime);
        observe(result, now() - started, fresh);
        if (fresh) publish(result, timestamp, now() - started);
      }
    } catch (error) {
      if (error.name === 'AbortError' || !active()) break;
      if (error.status !== 429) throw error;
    }
    await wait(Math.max(0, interval - (now() - started)));
  }
}
