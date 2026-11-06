"""Gain dans le stacking Ridge à 33 modèles (common.evaluate) des variantes en cache."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import evaluate
C = Path(__file__).resolve().parent / "cache"
ld = lambda n: np.load(C / f"{n}_oof.npy")
cands = {
    "base_s0": [ld("base")], "gb_s0": [ld("gb")],
    "base_3seeds": [ld("base"), ld("base@s1"), ld("base@s2")],
    "gb_3seeds": [ld("gb"), ld("gb@s1"), ld("gb@s2")],
    "gb+oafe_3seeds": [ld("gb+oafe"), ld("gb+oafe@s1"), ld("gb+oafe@s2")],
    "gb_wide_3seeds": [ld("gb_wide"), ld("gb_wide@s1"), ld("gb_wide@s2")],
    "gb_wide+oafe_3seeds": [ld("gb_wide+oafe"), ld("gb_wide+oafe@s1"), ld("gb_wide+oafe@s2")],
    "oafe_3seeds": [ld("oafe"), ld("oafe@s1"), ld("oafe@s2")],
}
for k in sys.argv[1:] or cands:
    r = evaluate(np.mean(cands[k], axis=0))
    print("STACK", k, json.dumps(r, ensure_ascii=False), flush=True)
