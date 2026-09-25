# BallTracker roadmap

## Current state (2026-09-25)

Two codebases live in this repo:

| Part | What it does | Entry point |
|---|---|---|
| `src/balltracker/` | Cricket bowling analytics: stump-anchored PnP camera calibration, HSV / motion / hybrid ball detection with Kalman + CSRT tracking, bounce-segmented gravity fit, metrics (release speed, swing, post-bounce turn, pitch line/length), plain-language summaries | `streamlit run scripts/run_session_app.py`, `scripts/analyze_delivery.py`, `scripts/calibrate_stumps.py` |
| `balltrace/` | Separate local-first UI: FastAPI backend sends frames to an Ollama vision model (`qwen3-vl:8b`) for detection; React/Vite frontend draws the trajectory overlay and exports JSON/CSV/MP4 | `uvicorn app.main:app --app-dir balltrace/backend`, `npm run dev` in `balltrace/frontend` |

What works, and how it was verified today:

| Check | Command | Result |
|---|---|---|
| Analytics package | `.venv/bin/python -m pytest tests/ -q` | 14 passed in 5.68s (synthetic-video end to end, PnP recovery, physics round trip, metrics keys, plain language, hybrid tracker with a mock release seed) |
| BallTrace backend | `cd balltrace/backend && pytest -q` | 43 passed in 0.30s (JSON parse/repair, window merge, trajectory gaps, normalized coords, timestamps, contact sheet). Needs no Ollama, ffmpeg or video |
| BallTrace frontend | `cd balltrace/frontend && npm run build` | `tsc -b && vite build` OK, 249.94 kB JS bundle |
| Lint | `.venv/bin/ruff check .` / `npm run lint` | All checks passed / 0 warnings after this commit |

Not verified by any automated check:

- Any path that talks to a real Ollama server: `OllamaClient` in `balltrace/backend/app/services/ollama_client.py` and `run_analysis` in `balltrace/backend/app/services/analysis_pipeline.py`.
- MediaPipe pose release seeding on real footage (`src/balltracker/release.py`); the only test asserts `mediapipe_available()` returns a bool.
- The Streamlit session app on real clips; `scripts/run_session_app.py` has no test.
- Accuracy against radar or measured bounce markers. Every accuracy test runs on synthetic deliveries from `src/balltracker/synthetic.py`.

## Known gaps and bugs

- No pre-commit hooks or CI. ruff and oxlint run only by hand.
- Oversized files: `scripts/run_session_app.py` (830 lines), `src/balltracker/tracking.py` (598). `track_delivery_hybrid` (`tracking.py:225` to `:432`) is one ~200-line function.
- Frontend has no test runner (`balltrace/frontend/package.json` has no test script); `src/lib/trajectory.ts`, `src/lib/status.ts` and the hooks are untested.
- `balltrace/API_CONTRACT.md` shows `window_size: 8, window_overlap: 4`, but `balltrace/backend/app/config.py` defaults are 4 / 0, `ANALYZE_FRAMES_INDIVIDUALLY = True` and `DEFAULT_FRAME_STRIDE = 3`. The contract does not document `frame_stride`.
- `requirements.txt` lists `ultralytics` unconditionally while `pyproject.toml` puts it in the optional `detection` extra. Nothing under `src/` or `scripts/` imports it.
- BallTrace backend shares the root `.venv` and `balltrace/backend/requirements.txt` is unpinned, so a fresh install is not reproducible.
- `data/ground_truth/template.csv` is a header line only; no labelled real delivery exists in the repo, so `evaluate_against_gt` in `src/balltracker/pipeline.py` only ever runs on synthetic data.
- `.venv` is Python 3.14 while `requires-python >= 3.10`; whether `mediapipe` and `ultralytics` install on 3.14 is not asserted anywhere.
- `useVideoPlayer` attached listeners via `[videoRef.current]` in the effect deps (fixed this commit with a callback ref; the linter's suggested `[]` would have missed the `<video>` element that mounts only after upload).

## Milestones

### M1. Guardrails: hooks and CI

- [ ] Add `.pre-commit-config.yaml` running `ruff` (root and `balltrace/backend`) and `oxlint` (`balltrace/frontend`); document `pre-commit install` in README.
- [ ] Pin `balltrace/backend/requirements.txt`; reconcile `requirements.txt` with the `pyproject.toml` extras (drop `ultralytics` or move it to the extra).
- [ ] Add a GitHub Actions workflow that runs root pytest, backend pytest, and frontend lint + build.

Done when: the workflow is green on `main` for all three jobs, and a local commit containing a ruff violation is rejected by the hook.

### M2. Validate on real footage

- [ ] Label at least 5 real deliveries with radar speed and a measured bounce marker per `docs/GROUND_TRUTH.md`, into `data/ground_truth/`.
- [ ] Add a pytest marked `real` that runs `analyze_delivery` on a labelled clip (skip if the clip is missing) and asserts release speed and bounce position within a stated tolerance.
- [ ] Write the measured error bands (speed, bounce) into the README status line with the date.

Done when: `pytest tests/ -q -m real` passes against the labelled set and the README states measured accuracy with a date.

### M3. Break up the oversized modules

- [ ] Split `scripts/run_session_app.py` into per-tab modules under `src/balltracker/app/` (calibrate, analyze, session review).
- [ ] Split `track_delivery_hybrid` into seed, gated-detect and CSRT-fallback functions, each under 75 lines.

Done when: `wc -l` shows no source file over 500 lines, and the 14 root and 43 backend tests still pass.

### M4. Test the Ollama path and the frontend

- [ ] Add a fake Ollama fixture (`httpx.MockTransport`) and test `run_analysis` end to end: per-frame mode, window mode, parse-failure retry, cancel.
- [ ] Add vitest with tests for `lib/trajectory.ts` and `lib/status.ts`; add `npm test`.
- [ ] Fix `API_CONTRACT.md` defaults and document `frame_stride`.

Done when: a coverage run shows `analysis_pipeline.run_analysis` executed by tests, and `npm test` runs in the M1 workflow.

### M5. Decide how BallTracker and BallTrace relate

- [ ] Either add an `"ollama"` detection method to `src/balltracker/detection.py` that consumes BallTrace detections and feeds the physics fit, or move `balltrace/` to its own repo.
- [ ] If merged: one install path and one quick start in the root README.

Done when: a single documented command analyses one real clip and yields both the overlay and the speed/bounce metrics, or the split is done and each README links to the other.

## Won't do / out of scope

- Cloud hosting, accounts, or multi-user sessions. Both apps are local-first by design (SQLite, local video files).
- YOLO / Detectron detection inside BallTrace. Its README states the Ollama vision model is the only detector; YOLO training lives in the separate `TeleVision05/BallTracking` repo.
- A mobile capture app.
- Non-standard pitch lengths (junior, women's) until M2 establishes accuracy on the standard men's pitch.

## Open questions for the owner

1. Is `TeleVision05/BallTracking` (YOLO trainer, October 2025) still active, or should it be archived and linked from here so the two names stop colliding?
2. Do you have radar-verified deliveries to label for M2? Without them every accuracy claim stays synthetic-only.
3. Should BallTrace stay in this repo (M5) or become its own?
4. Is `ultralytics` still needed? Nothing in `src/` or `scripts/` imports it.
5. Target Python: keep the 3.14 venv, or pin 3.11 / 3.12 for guaranteed `mediapipe` wheels?
