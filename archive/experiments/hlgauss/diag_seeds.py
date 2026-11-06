"""Diagnostic multi-graines sur le pli 0 : la diversité HL-Gauss vs MLP de référence dépasse-t-elle la simple
diversité entre deux graines du même MLP ?

Pour chaque config X (graines 0, 1, 2 dans preds/) :
- rmse : moyenne sur les graines (et écart-type) ;
- corr_intra : corrélation moyenne des résidus entre deux graines de X ;
- corr_ref : corrélation moyenne des résidus entre une graine de X et une graine de la référence ;
- moy_ref+X : RMSE de la moyenne (ref@a + X@b)/2, moyennée sur les couples (a ≠ b pour X = ref) ;
- ens3 : RMSE de la moyenne des 3 graines de X ; ens3+3 : moyenne des 3 graines de ref et des 3 de X.
"""

import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load, rmse  # noqa: E402

_, _, _, _, folds, y = load()
yv = y[folds == 0]
P = {}
for f in sorted((HERE / "preds").glob("*_fold0.npy")):
    name = f.stem.removesuffix("_fold0")
    cfg, _, off = name.partition("@")
    P.setdefault(cfg, {})[int(off or 0)] = np.load(f)

ref = P["ref"]


def c(a, b):
    return float(np.corrcoef(yv - a, yv - b)[0, 1])


print(f"{'config':10s} {'n':>2s} {'rmse':>14s} {'corr_intra':>10s} {'corr_ref':>9s} {'moy_ref+X':>9s} {'ens3':>8s} {'ens3+3':>8s}")
for cfg, d in P.items():
    s = sorted(d)
    r = [rmse(yv, d[k]) for k in s]
    intra = np.mean([c(d[a], d[b]) for a, b in itertools.combinations(s, 2)]) if len(s) > 1 else np.nan
    pairs = [(a, b) for a in ref for b in s if cfg != "ref" or a != b]
    cr = np.mean([c(ref[a], d[b]) for a, b in pairs])
    mix = np.mean([rmse(yv, (ref[a] + d[b]) / 2) for a, b in pairs])
    ens = rmse(yv, np.mean([d[k] for k in s], 0))
    ens33 = rmse(yv, np.mean([d[k] for k in s] + [ref[k] for k in ref], 0)) if cfg != "ref" else np.nan
    print(f"{cfg:10s} {len(s):2d} {np.mean(r):8.3f}±{np.std(r):.2f} {intra:10.4f} {cr:9.4f} {mix:9.3f} {ens:8.3f} {ens33:8.3f}")
