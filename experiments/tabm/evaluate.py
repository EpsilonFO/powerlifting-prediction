"""uv run python tabm/evaluate.py <nom> : evaluate() sur experiments/results/<nom>_oof.npy (processus sans torch)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402

from common import RESULTS, evaluate  # noqa: E402

name = sys.argv[1]
ev = evaluate(np.load(RESULTS / f"{name}_oof.npy"))
print(json.dumps(ev, indent=1))
(HERE / "logs" / f"{name}_evaluate.json").write_text(json.dumps(ev, indent=1))
