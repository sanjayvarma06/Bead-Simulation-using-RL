import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("our_model_code", "research_paper_code", "experiments"):
    sys.path.insert(0, str(ROOT / sub))
