"""Évalue ensemble les modèles neufs (results/*_oof.npy) : seuls, puis cumulés dans le stacking à 33."""
import sys
import numpy as np
from common import RESULTS, evaluate
names = sys.argv[1:]
O = {n: np.load(RESULTS / f"{n}_oof.npy") for n in names}
for n in names:
    e = evaluate(O[n]); print(n, e["rmse"], "delta", e["delta"], e["delta_plis"])
e = evaluate(O[names[0]], [O[n] for n in names[1:]])
print("ensemble", "+".join(names), "stack", e["stack_new"], "delta", e["delta"], e["delta_plis"])
