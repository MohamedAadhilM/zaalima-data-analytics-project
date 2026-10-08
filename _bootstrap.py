"""Locate the repository root so every week file runs from anywhere (IDLE, Jupyter, terminal)."""
import sys
from pathlib import Path

try:
    _here = Path(__file__).resolve().parent
except NameError:                       # Jupyter / interactive
    _here = Path.cwd()
ROOT = next(p for p in [_here, *_here.parents] if (p / "src" / "cpe").exists())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
