# ITAS-MM: Intelligent Multimodal Traffic Assistance System

Real-time traffic sign recognition, lane detection, and legally-grounded
driving guidance for a single forward-facing camera setup.

## Overview

ITAS-MM connects visual perception (traffic signs + lane position) with
road-zone context (Urban / Suburban / Highway) and a Retrieval-Augmented
Generation (RAG) layer over a traffic-regulation knowledge base, then
converts the resulting guidance into speech.

## Problem statement

Standalone sign/lane detection tells a driver *what* was seen, not *what it
means right now*. ITAS-MM adds the missing context layer (road zone) and the
missing knowledge layer (regulation retrieval) so the system can say, in
plain speech, why a detected sign or lane event matters in the current
situation.

## Architecture

Four independent, testable stages, connected by a pipeline orchestrator:

```
Camera + GPS -> Perception -> Context -> Reasoning -> Interaction -> Voice guidance
```

| Stage | Module | Responsibility |
|---|---|---|
| Perception | `src/perception` | Sign detection (YOLOv11 + SAHI), lane detection |
| Context | `src/context` | Maps GPS coordinates to a road-zone label |
| Reasoning | `src/reasoning` | RAG over a traffic-regulation corpus, grounded via a local LLM (Ollama) |
| Interaction | `src/interaction` | Converts guidance text to speech (offline TTS) |

## Current status (IMPORTANT — read before demoing)

This is a **scaffold**, not a trained system. Concretely:

- **Sign detector**: wraps `ultralytics` YOLOv11 + `sahi` correctly, but
  ships with **no trained weights**. You must train on a labeled sign
  dataset (e.g. GTSDB, GTSRB, Mapillary Traffic Sign, or your own) and point
  `configs/perception.yaml` at the resulting `.pt` file. Without weights,
  `SignDetector` falls back to a stub that returns an empty detection list
  and logs a warning — it does not fabricate detections.
- **Lane detector**: implemented as a **classical CV baseline**
  (Canny edge + probabilistic Hough transform), which actually runs without
  training. A learned lane detector (row-wise / structure-aware) is listed
  as future work in Limitations below.
- **Zone classifier**: rule-based radius lookup against
  `configs/context.yaml`. Actually runs, but the zones defined in the
  sample config are placeholder coordinates — replace with real zone
  boundaries for your test route.
- **RAG / Reasoning**: fully wired against a local Ollama instance
  (embeddings + generation), but `data/regulations/` ships with only a
  **small illustrative sample** of traffic-rule text, not a real corpus.
  You must supply the actual regulation documents.
- **TTS**: uses `pyttsx3` (offline, no API cost), works out of the box.

## Installation

```bash
git clone <this-repo>
cd itas-mm
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Ollama (for the Reasoning stage)

```bash
# install from https://ollama.com
ollama pull llama3.1          # generation model
ollama pull nomic-embed-text  # embedding model
ollama serve                  # if not already running
```

## Environment

- Python 3.10+
- GPU recommended for YOLOv11 training/inference (CPU works for the demo
  pipeline with the stub detector)
- Ollama running locally on `http://localhost:11434` (configurable in
  `configs/reasoning.yaml`)

## Project structure

```
itas-mm/
├── configs/          # YAML config per stage — never hard-code values in src/
├── src/
│   ├── perception/    # sign detection, lane detection
│   ├── context/         # zone classification
│   ├── reasoning/        # RAG: chunking, vector store, generation
│   ├── interaction/       # TTS
│   ├── pipeline/            # orchestrator wiring all stages
│   └── utils/                 # logging, config loader, shared types
├── backend/          # FastAPI app exposing the pipeline over HTTP
├── frontend/         # Minimal React (Vite) demo UI
├── data/             # raw/processed datasets, regulation corpus (gitignored)
├── tests/            # unit + integration tests
└── scripts/          # one-off utility scripts
```

## Running the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

## Running the frontend

```bash
cd frontend
npm install
npm run dev
```

## Training the sign detector

Not included in this scaffold (requires a labeled dataset). Standard
`ultralytics` training loop applies:

```bash
yolo detect train data=<your_data.yaml> model=yolo11n.pt epochs=100 imgsz=640
```

Once trained, set `sign_detector.weights_path` in `configs/perception.yaml`
to the resulting `best.pt`.

## Evaluation

Not yet implemented — no trained model exists to evaluate yet. Once the
sign detector is trained, add mAP/precision/recall/confusion-matrix
reporting against a held-out test split (see Limitations).

## Inference

`src/pipeline/orchestrator.py` exposes `Pipeline.run(image, gps_coords)`
which runs all four stages and returns a structured result, including the
raw guidance text and (optionally) synthesized audio bytes.

## Results

No results yet — no model has been trained and no real regulation corpus
has been loaded. This section will be filled in once training is complete;
reporting fabricated numbers here would violate the project's own
correctness-first principle.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `SignDetector` always returns `[]` | No weights configured — expected until you train a model |
| RAG stage raises a connection error | Ollama not running, or wrong `base_url` in `configs/reasoning.yaml` |
| TTS produces no audio on Linux | Missing `espeak`/`espeak-ng` system package (`pyttsx3` dependency) |
| Lane detector output looks noisy | Classical CV baseline is sensitive to lighting/road markings — expected limitation |

## Limitations

- No trained weights ship with this repo (see Current status)
- Lane detection is a classical-CV baseline, not the learned architecture
  described in the proposal (Stage 1 future work)
- Regulation corpus is illustrative only, not a complete legal database
- Zone boundaries in `configs/context.yaml` are placeholders
- No dataset-level evaluation has been run — do not treat this scaffold as
  validated for correctness

## Future improvements

- Train and evaluate YOLOv11 sign detector on a real dataset with a proper
  train/val/test split and leakage checks
- Replace classical-CV lane detector with a learned row-wise/structure-aware
  model
- Load a real, versioned traffic-regulation corpus with citations
- Add CI (lint + tests) and containerization (Dockerfile)

## License

Add your chosen license here (not specified in the project registration).
