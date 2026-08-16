"""conftest for synthetic_trial tests — adds src/ to the import path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
