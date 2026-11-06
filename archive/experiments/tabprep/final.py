"""Livrable : meilleure variante TabPrep (GroupBy relatifs larges + OAFE, LightGBM tuné, moyenne de 3 seeds, 5 plis)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import evaluate, save
C = Path(__file__).resolve().parent / "cache"
def avg(v):
    names = [v, f"{v}@s1", f"{v}@s2"]
    return (np.mean([np.load(C / f"{n}_oof.npy") for n in names], 0),
            np.mean([np.load(C / f"{n}_test.npy") for n in names], 0))
oof, te = avg("gb_wide+oafe")
save("tabprep_lgbm", oof, te)
print("FINAL tabprep_lgbm", json.dumps(evaluate(oof), ensure_ascii=False))
oof2, te2 = avg("gb")
save("tabprep_gb_lgbm", oof2, te2)  # variante secondaire (GroupBy fidèle seul), évaluée dans stack_eval.py
