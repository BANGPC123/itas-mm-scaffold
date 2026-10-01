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
  (embeddings + generation) and the three-source regulation corpus described
  below. Retrieved evidence retains document and locator metadata; benchmark
  results are a retrieval snapshot, not a legal-quality guarantee.
- **TTS**: uses `pyttsx3` (offline, no API cost), works out of the box.

## Installation

```bash
git clone <this-repo>
cd itas-mm
conda create -n WTF python=3.12 -y
conda activate WTF
python -m pip install -r requirements.txt
```

Reference Windows GPU setup for this project uses PyTorch `2.14.0+cu130`
with CUDA 13.0. After installing the project requirements, install the CUDA
build and restore the project's NumPy constraint:

```bash
python -m pip install --force-reinstall torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install "numpy>=1.26,<2.0"
```

### Ollama (for the Reasoning stage)

```bash
# install from https://ollama.com
ollama pull llama3.1          # generation model
ollama pull nomic-embed-text  # embedding model
ollama serve                  # if not already running
```

## Environment

- Reference development environment: Conda `WTF`, Python 3.12.
- Reference GPU runtime: NVIDIA RTX 4070 Laptop GPU, PyTorch `2.14.0+cu130`,
  CUDA 13.0. `configs/perception.yaml` targets sign inference at `cuda:0`.
- Ollama must provide `llama3.1` for generation and `nomic-embed-text` for
  embeddings. Model names and base URL come from `configs/reasoning.yaml`.
- Backend host/port come from the Uvicorn launch command. `LOG_LEVEL` is the
  only optional process environment variable currently read by the app.
- The application does not load a `.env`/dotenv file.

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

## Regulation corpus

Build the legal corpus and its derived Chroma index in this order:

```bash
python scripts/fetch_regulations.py
python scripts/build_vector_index.py
```

The Selective Port v1 scope is limited to Law 36/2024/QH15, Decree
168/2024/NÄ-CP, and QCVN 41:2024/BGTVT. QCVN sign-image pages are not OCRed;
the corpus never uses model knowledge to repair legal text. Raw source files
remain immutable provenance, while Chroma is derived and can be safely rebuilt
from `data/regulations/corpus.json`. The corpus port is attributed to
`lqb464/LuatRAG@ae2b1c796503e2a58493771bc341b66fb488e053`.

Run the fixed 24-case retrieval benchmark against an existing local Chroma
index and Ollama embedding service with:

```bash
conda run --no-capture-output -n WTF python -X utf8 scripts/evaluate_legal_retrieval.py
```

### Retrieval benchmark snapshot (2026-10-01)

The command above was run against the existing local index with corpus
fingerprint `51d2da86002c4757fcd9fffb2849447b2bd1805f86242d67f0c003f7ee78f9db`,
schema version `1`, and embedding model `nomic-embed-text`, at top-k `4`.
Observed results: 24 cases, Recall@1 `0.125000`, Recall@3 `0.291667`,
Recall@4 `0.333333`, and MRR `0.197917`.

The top-4 misses were `speed-01`, `speed-02`, `signals-01`, `signals-02`,
`prohibitory-signs-01`, `prohibitory-signs-03`, `lanes-01`,
`stopping-parking-01`, `stopping-parking-02`, `stopping-parking-03`,
`overtaking-01`, `penalties-01`, `penalties-02`, `penalties-03`,
`qcvn-sign-meaning-01`, and `qcvn-sign-meaning-02`. This is a reproducible
retrieval snapshot only; it is not a legal-quality guarantee or a quality gate.

## Inference

`src/pipeline/orchestrator.py` exposes
`Pipeline.run(image, latitude, longitude, synthesize_audio=True)`. It runs
all four stages and returns a `PipelineResult` containing perception,
context, grounded reasoning evidence, and an optional synthesized-audio
file path in `audio_path`.

## Results

No perception-model results yet — no model weights have been trained. The
retrieval snapshot above is reported separately from perception evaluation.

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
