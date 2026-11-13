"""``python -m balltracker`` runs the same command line as ``balltracker``."""

import sys

from .cli import main

sys.exit(main())
