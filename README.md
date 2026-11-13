# BallTracker

[![CI](https://github.com/TeleVision05/ball-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/TeleVision05/ball-tracker/actions/workflows/ci.yml)

Cricket bowling analytics from behind-the-stumps video: stump-anchored camera calibration, ball detection and tracking, a bounce-segmented physics fit, and metrics (release speed, swing, post-bounce turn, pitch line and length) with plain-language summaries.

**Status: working.** CI runs the tests, lint and builds on every pull request and push to `main` (see the badge above). Phone captures can be analysed from their sidecar. Accuracy is validated on synthetic deliveries only. See [ROADMAP.md](ROADMAP.md).

> **BallTrace** (a separate local-first UI that detects the ball with an Ollama vision model) lives in [`balltrace/`](balltrace/README.md). It has its own README.

## Requirements

- Python 3.11 or 3.12
- For BallTrace only: Node.js 18+, [Ollama](https://ollama.com) with `qwen3-vl:8b`, optional FFmpeg

## Install

```bash
cd BallTracker
python3.12 -m venv .venv   # or python3.11
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This editable install with the `dev` extra is the only install path. `requirements.txt` points at it, and `pyproject.toml` holds every dependency and version bound.

To run the commit checks locally, install the hooks once: `pre-commit install`. They run ruff (check and format) on all Python and oxlint on `balltrace/frontend`; `pre-commit run --all-files` runs them on everything. The frontend hook needs `npm ci` in `balltrace/frontend` first.

## Configure

There are no environment variables and no `.env` file.

- Pitch geometry, camera, detection and physics parameters: [`config/default.yaml`](config/default.yaml)
- BallTrace backend constants (Ollama URL, model, frame stride, timeouts): [`balltrace/backend/app/config.py`](balltrace/backend/app/config.py); Ollama URL and model can also be changed in the UI Settings panel and are stored in SQLite.

## Run

1. Drop `.mp4` / `.mov` clips into `data/raw/real/` (gitignored; see [data/raw/real/README.md](data/raw/real/README.md)).
2. Calibrate and analyze in the Streamlit app:

   ```bash
   streamlit run scripts/run_session_app.py
   ```

   **Calibrate** tab: load a frame, click the stump landmarks in order, save. **Analyze** tab: run analysis, view pitch maps and overlays.

3. CLI alternative:

   ```bash
   python scripts/calibrate_stumps.py --video data/raw/real/clip.mp4 --out data/raw/real/session.calib.json
   python scripts/analyze_delivery.py --video data/raw/real/clip.mp4 --calib data/raw/real/session.calib.json --out /tmp/analysis.json
   ```

4. Phone clips are JSON sidecars from the Android capture app:

   ```bash
   python scripts/analyze_capture.py --sidecar path/to/clip.json --out /tmp/analysis.json
   ```

   The video is the `videoFile` next to the sidecar. `--method hsv` tracks a red ball, including the synthetic clip. `--method white` tracks a white ball. The default `--method auto` uses the hybrid motion tracker.

Filming and calibration guidance: [docs/CAPTURE_PROTOCOL.md](docs/CAPTURE_PROTOCOL.md). Delivery label schema: [docs/GROUND_TRUTH.md](docs/GROUND_TRUTH.md).

## Test

```bash
pytest tests/ -q                                 # analytics package
cd balltrace/backend && pytest -q                # BallTrace backend
cd balltrace/frontend && npm run lint && npm run build
ruff check . && ruff format --check .
```

Backend tests need no Ollama, ffmpeg or video file.

## Method (v1)

1. Stump-anchored camera pose (PnP) in world pitch coordinates
2. Ball detect and track (HSV / motion / hybrid with optional MediaPipe release seed and CSRT)
3. Bounce-segmented gravity fit (optional drag) with reprojection and ray distance
4. Metrics: release speed, swing, post-bounce turn, pitch line/length

`src/balltracker/synthetic.py` simulates deliveries and renders synthetic video for the tests; no demo footage is required.

## Layout

```
config/default.yaml        pitch, camera, detection, physics, paths
docs/                      CAPTURE_PROTOCOL.md, GROUND_TRUTH.md
data/ground_truth/         label schema and CSV template
scripts/                   calibrate_stumps.py, analyze_delivery.py, run_session_app.py
src/balltracker/           package source
tests/                     pytest suite
balltrace/                 BallTrace backend (FastAPI) and frontend (React/Vite)
```

## License

[MIT](LICENSE), copyright 2026 Tarun Vadapalli. The licence covers the code only. Video footage and capture sessions are not part of the repository and must never be committed.
