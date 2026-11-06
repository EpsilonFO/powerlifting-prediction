"""Témoins : gain dans le stacking de modèles 5 plis existants hors des 33 (le gain de TabM vient-il seulement
du fait d'être entraîné sur un autre découpage ?), et TabM + MLP te 1 seed."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from common import evaluate, OOF_DIR, RESULTS
out = {}
for n in sys.argv[1:]:
    f = RESULTS / f"{n}_oof.npy"
    oof = np.load(f if f.exists() else OOF_DIR / f"{n}.npy")
    ev = evaluate(oof)
    out[n] = ev
    print(n, json.dumps(ev), flush=True)
(HERE / "logs" / "controls.json").write_text(json.dumps(out, indent=1))
