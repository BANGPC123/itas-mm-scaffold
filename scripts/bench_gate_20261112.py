"""Queue gate: keep the GPU free until a wall-clock time, then let the queue continue.

Used as a "script" job in output/bench_20261112/queue.json. Holds no GPU memory while waiting.
The resume time is read from <bench>/gate_resume_at.txt ("YYYY-MM-DD HH:MM", local time) on every
poll, so it can be edited while the gate is waiting. Writes results.json into --out when released.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-file", type=Path, required=True)
    parser.add_argument("--default-resume", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.resume_file.is_file():
        args.resume_file.write_text(args.default_resume + "\n", encoding="utf-8")
    started = datetime.now()
    while True:
        text = args.resume_file.read_text(encoding="utf-8").strip()
        target = datetime.strptime(text, "%Y-%m-%d %H:%M")
        if datetime.now() >= target:
            break
        time.sleep(30)
    result = {"status": "COMPLETED", "gate_started": started.isoformat(timespec="seconds"),
              "resume_at": text, "released": datetime.now().isoformat(timespec="seconds")}
    (args.out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
