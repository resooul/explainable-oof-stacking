"""Makes the helpers in tests/ (e.g. synthetic.py) importable from every test module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
