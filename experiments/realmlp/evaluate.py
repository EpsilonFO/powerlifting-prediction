"""Évaluation d'une OOF RealMLP complète (5 plis), dans un processus sans PyTorch.

    uv run python realmlp/evaluate.py results/realmlp_oof.npy
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import evaluate  # noqa: E402

if __name__ == "__main__":
    oof = np.load(HERE.parent / sys.argv[1])
    print(json.dumps(evaluate(oof), indent=1, ensure_ascii=False))
