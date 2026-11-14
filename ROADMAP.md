# BallTracker roadmap

## Current state

Three codebases live in this repo:

| Part | What it does | Entry point |
|---|---|---|
| `src/balltracker/` | Cricket bowling analytics: stump-anchored PnP camera calibration, HSV / motion / hybrid ball detection with Kalman + CSRT tracking, bounce-segmented gravity fit, metrics (release speed, swing, post-bounce turn, pitch line/length), plain-language summaries | `streamlit run scripts/run_session_app.py`, `scripts/analyze_delivery.py`, `scripts/calibrate_stumps.py` |
| `android/` | Kotlin and Compose app that records high-speed clips with a JSON sidecar, analysed by `scripts/analyze_capture.py` | `./gradlew :app:assembleDebug` in `android/` |
| `balltrace/` | Separate local-first UI: FastAPI backend sends frames to an Ollama vision model (`qwen3-vl:8b`) for detection; React/Vite frontend draws the trajectory overlay and exports JSON/CSV/MP4 | `uvicorn app.main:app --app-dir balltrace/backend`, `npm run dev` in `balltrace/frontend` |

How to check that it works. The CI badge in the README shows the latest result, so this table holds commands, not counts:

| Check | Command | What it covers |
|---|---|---|
| Analytics package | `.venv/bin/python -m pytest tests/ -q` | Synthetic-video end to end, PnP recovery, physics round trip, metrics keys, plain language, hybrid tracker with a mock release seed, and synthetic stand-ins for the defects found on real footage (`tests/test_real_footage_regressions.py`: click-gap fill, lossless upright copy, result checks) |
| BallTrace backend | `cd balltrace/backend && pytest -q` | JSON parse/repair, window merge, trajectory gaps, normalized coords, timestamps, contact sheet, API contract defaults, Ollama client and `run_analysis` against a fake Ollama (success, model missing, server down, malformed reply, timeout, cancel), friendly API errors, upload limit. Needs no Ollama, ffmpeg or video |
| BallTrace frontend | `cd balltrace/frontend && npm test && npm run build` | Friendly-error helper, status and trajectory helpers, upload flow, results and error rendering, player and progress hooks; type check and production build |
| Lint | `.venv/bin/ruff check .` / `npm run lint` | Python lint and format, frontend lint (also run by the pre-commit hooks) |

Not verified by any automated check:

- Behaviour against a real Ollama server and a real vision model. The client and pipeline are tested only against a fake server (`httpx.MockTransport`); the real model's answers, speed and timeouts are not covered.
- MediaPipe pose release seeding on real footage (`src/balltracker/release.py`); the only test asserts `mediapipe_available()` returns a bool. Tried by hand on two stock clips on 2026-11-13: inside the delivery window it found the release frame to within two frames of the eye ([docs/REAL_FOOTAGE.md](docs/REAL_FOOTAGE.md)).
- `scripts/analyze_capture.py` on a real phone capture. None exists yet (2026-11-13).
- The Streamlit session app on real clips. It is driven headlessly on a synthetic clip (`tests/test_session_app.py`, Streamlit AppTest), but clicks on the picture come from a custom component AppTest cannot click, so the tests place points and marks through session state.
- Accuracy against radar or measured bounce markers. Every accuracy test runs on synthetic deliveries from `src/balltracker/synthetic.py`.

## Known gaps and bugs

- Oversized files: ~~`scripts/run_session_app.py` (830 lines), `src/balltracker/tracking.py` (598)~~ split on 2026-11-13 into `src/balltracker/app/` and `tracking_core.py` / `tracking_hybrid.py` / `tracking_manual.py`. Still over 500 lines: `android/.../camera/CameraController.kt` (817). Python functions still over 75 lines (AST count, 2026-11-13): ~~`pipeline.analyze_delivery` (120), `physics._fit_segment` (102), `physics.fit_trajectory` (101), `pipeline.analyze_from_track` (97)~~ under 75 since the real-footage change (shared `pipeline._delivery_result`, `physics._search_space` / `_snap_bounce_to_pitch` / `_fit_post_bounce`), `viz.render_overlay_video` (96), `export_service.export_trajectory_video` (90), `detection._score_motion_candidates` (85), `detection.detect_video` (85), `calibrate_stumps._interactive_points` (80), ~~`capture_analysis.analyze_capture` (77)~~, `release._wrist_track_tasks` (77), `viz.draw_pitch_map` (77).
- **Real footage gives no believable result yet** (2026-11-13, [docs/REAL_FOOTAGE.md](docs/REAL_FOOTAGE.md)). 26 runs on the only two real clips (24 fps stock nets footage, no phone capture exists) finished 17 times and refused 9 times, with no crash; none of the 17 results is believable. Fixed in that change: the click-gap filler that put balls behind the last click, the lossy copy made when turning a phone clip upright, and the silence (results now carry plain-language `checks`). Still open, each seen on those clips:
  - One `session.calib.json` per folder is reused for every clip in it (`pipeline._find_adjacent_calib_file`, the app sidebar), so a clip can be analysed with another clip's camera. Clip A's saved session was. No check can catch this; a calibration should name the clip and frame it came from.
  - Nothing detects a moving or cut camera. Clip A is handheld with cuts; clip B pans about 25 px between the calibration frame and the release.
  - No way to give the true capture rate when a file plays slower than it was filmed (slow-motion exports). The pipeline uses the container rate; the checks now warn below 100 fps.
  - Ball detection on real nets footage: at 900×474 the ball is 2 to 5 px and no detector produced a track the fit could use. In 11 of 15 runs the fit stopped on its slowest allowed release (about 29 km/h), which both saved app sessions had reported as a speed. Needs real 120 fps captures to work on.
- `data/ground_truth/template.csv` is a header line only; no labelled real delivery exists in the repo, so `evaluate_against_gt` in `src/balltracker/pipeline.py` only ever runs on synthetic data.
- `useVideoPlayer` attached listeners via `[videoRef.current]` in the effect deps (fixed this commit with a callback ref; the linter's suggested `[]` would have missed the `<video>` element that mounts only after upload).

## Milestones

### M1. Guardrails: hooks and CI

- [x] Add `.pre-commit-config.yaml` running `ruff` (root and `balltrace/backend`) and `oxlint` (`balltrace/frontend`); document `pre-commit install` in README.
- [x] Bound `balltrace/backend/requirements.txt`; reconcile `requirements.txt` with `pyproject.toml` (`ultralytics` stays only in the optional `detection` extra).
- [x] Add a GitHub Actions workflow that runs root pytest, backend pytest, and frontend lint + test + build.

Done when: the workflow is green on `main` for all three jobs, and a local commit containing a ruff violation is rejected by the hook.

### M2. Validate on real footage

First contact, 2026-11-13: the pipeline was run on the only real clips available (two stock clips, no phone capture) and produced nothing believable; see [docs/REAL_FOOTAGE.md](docs/REAL_FOOTAGE.md). The open items it found are under "Known gaps and bugs".

- [ ] Record at least 5 deliveries with the Android app per `docs/CAPTURE_PROTOCOL.md` and run `scripts/analyze_capture.py` on them.
- [ ] Label at least 5 real deliveries with radar speed and a measured bounce marker per `docs/GROUND_TRUTH.md`, into `data/ground_truth/`.
- [ ] Add a pytest marked `real` that runs `analyze_delivery` on a labelled clip (skip if the clip is missing) and asserts release speed and bounce position within a stated tolerance.
- [ ] Write the measured error bands (speed, bounce) into the README status line with the date.

Done when: `pytest tests/ -q -m real` passes against the labelled set and the README states measured accuracy with a date.

### M3. Break up the oversized modules

- [x] Split `scripts/run_session_app.py` into per-step modules under `src/balltracker/app/` (setup, analyze, mark, result, plus `logic` and `images` for the non-UI parts), behind headless app tests.
- [x] Split `track_delivery_hybrid` into seed, gated-detect and CSRT-fallback functions, each under 75 lines, behind characterisation tests and exact pins (`tests/test_tracker_pins.py`).
- [ ] Split `android/.../camera/CameraController.kt` (817 lines).

Done when: `wc -l` shows no source file over 500 lines, and the root and backend test suites still pass.

### M4. Test the Ollama path and the frontend

- [x] Add a fake Ollama fixture (`httpx.MockTransport`) and test `run_analysis` end to end: per-frame mode, window mode, parse-failure retry, cancel (`balltrace/backend/tests/test_analysis_pipeline.py`, `test_ollama_client.py`).
- [x] Add vitest with tests for `lib/trajectory.ts` and `lib/status.ts`; add `npm test` (and the friendly-error helper, upload flow, results rendering and hooks).

Done when: a coverage run shows `analysis_pipeline.run_analysis` executed by tests, and `npm test` runs in the M1 workflow.

### M5. Decide how BallTracker and BallTrace relate

- [ ] Either add an `"ollama"` detection method to `src/balltracker/detection.py` that consumes BallTrace detections and feeds the physics fit, or move `balltrace/` to its own repo.
- [ ] If merged: one install path and one quick start in the root README.

Done when: a single documented command analyses one real clip and yields both the overlay and the speed/bounce metrics, or the split is done and each README links to the other.

## Won't do / out of scope

- Cloud hosting, accounts, or multi-user sessions. Both apps are local-first by design (SQLite, local video files).
- YOLO / Detectron detection inside BallTrace. Its README states the Ollama vision model is the only detector; YOLO training lives in the separate `TeleVision05/BallTracking` repo.
- Non-standard pitch lengths (junior, women's) until M2 establishes accuracy on the standard men's pitch.

## Open questions for the owner

1. Is `TeleVision05/BallTracking` (YOLO trainer, October 2025) still active, or should it be archived and linked from here so the two names stop colliding?
2. Do you have radar-verified deliveries to label for M2? Without them every accuracy claim stays synthetic-only.
3. Should BallTrace stay in this repo (M5) or become its own?
4. Is `ultralytics` still needed? Nothing in `src/` or `scripts/` imports it.
