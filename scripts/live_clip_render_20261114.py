"""M12: render a 30 s live-demo clip starting at a given second (same parameters as baseline_render_20261114.py)."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.render_offline_video_max_20261014 import render
from src.pipeline.orchestrator import Pipeline


def main():
    src, out, start = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 30.
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = render(src, out / 'render', analysis_hz=6., pipeline=Pipeline(), output_fps_cap=30, lane_every=2,
                    max_seconds=seconds, start_second=start)
    wall = time.perf_counter() - t0
    (out / 'baseline_meta.json').write_text(json.dumps({'source': str(src), 'start_sec': start,
        'wall_seconds': round(wall, 2), 'result': result}, ensure_ascii=False, default=str, indent=2), encoding='utf-8')
    print('LIVE_DONE', out, round(wall, 2), flush=True)


if __name__ == '__main__':
    main()
