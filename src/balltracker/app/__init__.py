"""The Streamlit session app, split by step.

Launch it with ``streamlit run scripts/run_session_app.py``. Layout:

- ``main``: page header, paths, step routing and the step picker.
- ``sidebar``: video folder, upload and session loading.
- ``setup_step``: step 1, clicking stump points to calibrate the camera.
- ``analyze_step``: step 2, trimming the clip and automatic analysis.
- ``mark_step``: step 2 in marking mode, clicking the ball frame by frame.
- ``result_step``: step 3, speed, pitch map and replay.
- ``state``: session-state defaults and navigation helpers.
- ``logic`` and ``images``: everything that does not need Streamlit, tested
  directly in ``tests/test_app_logic.py``.
"""
