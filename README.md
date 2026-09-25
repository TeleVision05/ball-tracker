# BallTracker

Cricket bowling analytics package for behind-stumps video capture, ball-path reconstruction, and ground-truth validation against radar and measured bounce markers.

> **BallTrace** (Ollama vision ball tracking UI) lives in [`balltrace/`](balltrace/README.md) — React + FastAPI, local-first trajectory overlay. See that README to run it.

## Features

- Standard men's cricket pitch geometry with 3D stump landmarks
- Interactive stump calibration for real user-provided footage
- Ball detect + track with configurable detection method (auto / HSV / motion)
- Capture protocol and ground-truth schema documentation
- Configurable physics, camera, and detection parameters

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

## Install

```bash
cd BallTracker
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Or without dev extras:

```bash
pip install -r requirements.txt
pip install -e .
```

## Primary workflow (real videos)

1. **Drop videos** into `data/raw/real/` (`.mp4` / `.mov`).

2. **Calibrate** in the UI (recommended) or CLI:

   ```bash
   streamlit run scripts/run_session_app.py
   ```

   Use the **Calibrate** tab: load a frame, click stump landmarks in order, save.

   CLI alternative:

   ```bash
   python scripts/calibrate_stumps.py \
     --video data/raw/real/your_clip.mp4 \
     --out data/raw/real/session.calib.json
   ```

3. **Analyze** via the **Analyze** tab, or:

   ```bash
   python scripts/analyze_delivery.py \
     --video data/raw/real/your_clip.mp4 \
     --calib data/raw/real/session.calib.json \
     --out /tmp/analysis.json
   ```

4. **Or use the session app** for batch analysis, pitch maps, and overlays:

   ```bash
   streamlit run scripts/run_session_app.py
   ```

See [data/raw/real/README.md](data/raw/real/README.md) and [docs/CAPTURE_PROTOCOL.md](docs/CAPTURE_PROTOCOL.md) for filming and calibration guidance.

## Method (v1)

1. Stump-anchored camera pose (PnP) in world pitch coordinates  
2. Ball detect + track (HSV / motion / auto)  
3. Bounce-segmented gravity/drag 3D fit with reprojection + ray distance  
4. Metrics: release speed, β_S / C_S swing, post-bounce turn, pitch line/length  

## Developer utilities

`synthetic.py` provides in-memory physics simulation and optional synthetic video generation for **unit tests and algorithm validation**. Demo video batches are not required for normal use — drop your own real clips instead.

Run tests:

```bash
pytest tests/ -q
```

Verify pitch geometry:

```bash
python -c "from balltracker.pitch import stump_landmarks_3d; print(len(stump_landmarks_3d()))"
```

## Project layout

```
BallTracker/
├── config/default.yaml      # Pitch, camera, physics, paths
├── docs/
│   ├── CAPTURE_PROTOCOL.md  # FullTrack-class capture setup
│   └── GROUND_TRUTH.md      # Delivery label schema
├── data/
│   ├── raw/real/            # Your delivery videos + session.calib.json
│   └── ground_truth/        # Schema and CSV template
├── scripts/
│   ├── calibrate_stumps.py  # Interactive stump calibration
│   ├── analyze_delivery.py
│   └── run_session_app.py
├── sessions/                # Saved session UX outputs
└── src/balltracker/         # Package source
```

## Documentation

- [Capture protocol](docs/CAPTURE_PROTOCOL.md) — camera placement, FPS, calibration, real-video workflow
- [Ground truth schema](docs/GROUND_TRUTH.md) — delivery annotation fields

## License

Private / research use.
