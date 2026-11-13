"""M0 baseline: render clips with the web parameters (analysis 6 Hz, lane_every=2, output_fps_cap=30)."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.render_offline_video_max_20261014 import render
from src.pipeline.orchestrator import Pipeline

def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)  # render() needs a NEW subdir
    t0 = time.perf_counter()
    result = render(src, out / "render", analysis_hz=6., pipeline=Pipeline(), output_fps_cap=30, lane_every=2)
    wall = time.perf_counter() - t0
    (out / 'baseline_meta.json').write_text(json.dumps({
        'source': str(src), 'wall_seconds': round(wall, 2),
        'result': result}, ensure_ascii=False, default=str, indent=2), encoding='utf-8')
    print('BASELINE_DONE', out, round(wall, 2), flush=True)

if __name__ == '__main__':
    main()
