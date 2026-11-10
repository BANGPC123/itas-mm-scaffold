# ITAS-MM integration update — 11 October 2026

The working Windows project at `D:\itas-mm` already contained trained sign
weights, UFLD-v2, temporal tracking, a 600-frame Vietnam labeling interface,
and supervised CULane experiments. This update preserves that work and brings
in the legal corpus and evidence handling from Task 6.

## Implemented in this update

- Offline BM25 retrieval over all three Task 6 document families, retaining
  exact document IDs, locators, excerpts, source URLs and SHA256 provenance.
  Search no longer depends on an Ollama server or readable decree PDF pages.
- Main sign / vehicle panel scope keeps the primary scope and intersects
  multiple reviewed vehicle conditions. `motorbike` and `motorcycle` match.
- Reviewed S.508a/b time intervals support multiple windows and midnight
  crossing. P.131b/c uses odd/even **Vietnam local calendar days**. These need
  an explicit timezone-aware observation datetime; relative video timestamps
  and server time never substitute for it. Interval ends are exclusive.
- The review UI accepts vehicle and time conditions. It does not infer the
  meaning of an unreadable plate or mark model output as human verification.
- API responses preserve label verification, context verification, placement,
  lane scope and time conditions. Task 6 evidence serializes as excerpts.
- Still-image overlays now use the same lane display-quality filter as video.
  Raw geometry remains separate from display smoothing; missing lanes are
  not extrapolated. Rejection messages distinguish geometric rejection from
  low-information images.
- Core software checks run in CI without distributing datasets or weights.

## Validation

On the working Windows project: **370 Python tests passed**, **7 JavaScript
tests passed**, and the Vite production build succeeded. Source/locator-aware
retrieval was evaluated on the unchanged **24-query Task 6 regression set**:

| Metric | Result |
|---|---:|
| Recall@1 | 50.00% |
| Recall@3 | 79.17% |
| Recall@4 | 83.33% |
| MRR | 0.6493 |

The corpus SHA256 is
`3e8f6b42f3077e77f20276c25d03df5e04a392131459b9103d0129ad2c273f19`.
A fresh HTTP backend also processed frames at 5s and 30s from the original
`test_video.mp4` with the installed models: UFLD-v2 returned structured output,
verification fields survived serialization, and legal search returned four
source-linked excerpts. The cold HTTP request took 8.74s including loading;
the next request took 330ms (300ms reported pipeline time). These two samples
are a smoke check with TTS disabled, not a throughput or accuracy benchmark.

These are retrieval regression metrics, not proof of legally correct driving
guidance, unseen-query generalization or an independent field TEST.

## Running on the existing Windows installation

```powershell
cd D:\itas-mm
.\start.ps1
```

Open `http://127.0.0.1:8000`. If an older backend is already running, restart
that backend to load the updated Python code. The frontend is already built.
The installed weights, datasets and rendered videos remain in place.

```powershell
$env:PYTHONPATH='D:\itas-mm\.runtime-packages;D:\itas-mm'
python -m pytest tests -q
node --test frontend/src/*.test.js
python scripts/evaluate_offline_retrieval.py
```

## Work still requiring new evidence

Vietnam lane training still needs reviewed pixel/polyline labels. The existing
600-frame dataset contains AI drafts; its internal held-out segment is from
the same development video. The uploaded 24-frame review export confirms one
frame; it does not provide enough labeled lane geometry for Vietnam training.

The supplied UFLD-v2 checkpoint and SegFormer CULane candidate are preserved.
No new Vietnam-trained model, CLRNet benchmark, field lane F1 or false-positive
rate is claimed by this update. Independent route videos, reviewed sign/lane
and supplementary-text labels, and source-disjoint evaluation are still needed.

`validated_corpus: false` remains in effect. Retrieved official text does not
establish that all amendments, exceptions, route scope or penalty conditions
are covered. Automatic OCR/vehicle/time interpretation of supplementary plates
also remains unvalidated. This is a functional research demo with measured
software checks; full proposal validation is still incomplete.
