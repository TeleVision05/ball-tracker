#!/usr/bin/env python3
"""Streamlit session app: 3-step senior-friendly bowl analysis flow.

Run with ``streamlit run scripts/run_session_app.py``. The app itself lives in
``balltracker.app`` (one module per step); this launcher makes sure the live
source tree is used and passes the project root, which holds the default
video folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Prefer the live source tree over any stale installed egg.
for _name in list(sys.modules):
    if _name == "balltracker" or _name.startswith("balltracker."):
        del sys.modules[_name]

from balltracker.app.main import main

main(ROOT)
