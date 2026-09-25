Drop delivery videos here (`.mp4` / `.mov`).

**Recommended:** `streamlit run scripts/run_session_app.py`
1. Upload or place videos in this folder
2. Open the **Calibrate** tab → load a frame → click stump landmarks in order → **Save calibration**
3. Open the **Analyze** tab → run analysis

CLI alternative:
```bash
python scripts/calibrate_stumps.py --video ... --out data/raw/real/session.calib.json
python scripts/analyze_delivery.py --video ... --calib data/raw/real/session.calib.json --out /tmp/out.json
```

