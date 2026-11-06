"""Lance TabM (ou le MLP de référence) sur des plis donnés.

    cd experiments && caffeinate -i uv run python tabm/run.py <nom> '<json params>' [plis] [--save] [--mlp]

Ex. : uv run python tabm/run.py ple_default '{}' 0
      uv run python tabm/run.py mlp_ref '{}' 0 --mlp
Écrit une ligne de résumé dans tabm/logs/summary.tsv.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

torch.set_num_threads(2)

from common import ROOT, cv_loop, evaluate, fold_rmse, rmse, save, load  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
name, params = args[0], json.loads(args[1]) if len(args) > 1 else {}
folds_arg = [int(k) for k in args[2].split(",")] if len(args) > 2 else None
do_save, use_mlp = "--save" in sys.argv, "--mlp" in sys.argv

if use_mlp:
    import models

    torch.set_num_threads = lambda n: None  # mlp_fn force 8 threads : on reste à 2
    mlp_json = json.loads((ROOT / "outputs" / "params" / "mlp.json").read_text())
    fn, params = models.mlp_fn, {**mlp_json, "te": True, **params}
else:
    from tabm_model import tabm_fn

    fn = tabm_fn

print(f"== {name} {params} plis={folds_arg}", flush=True)
t0 = time.time()
oof, test = cv_loop(fn, params, only_folds=folds_arg)
dt = (time.time() - t0) / 60
_, _, _, _, folds, y = load()
if folds_arg is None:
    res = {"rmse": round(rmse(y, oof), 3), "plis": fold_rmse(oof)}
else:
    res = {"plis": {k: round(rmse(y[folds == k], oof[folds == k]), 3) for k in folds_arg}}
print(f"== {name} : {res}  {dt:.1f} min", flush=True)
(HERE / "logs").mkdir(exist_ok=True)
with open(HERE / "logs" / "summary.tsv", "a") as f:
    f.write(f"{name}\t{json.dumps(params)}\t{folds_arg}\t{json.dumps(res)}\t{dt:.1f}\n")
if do_save and folds_arg is None:
    save(name, oof, test)  # évaluation dans un processus séparé : tabm/evaluate.py <nom>
