"""RealMLP (Holzmüller et al., NeurIPS 2024, pytabkit) dans la boucle CV du projet, sans toucher au projet.

Usage (depuis experiments/) :
    uv run python realmlp/run.py --name A_te --target A --te 1 --folds 0
    uv run python realmlp/run.py --name A_te --target A --te 1 --save realmlp     # 5 plis + save()
Options pytabkit supplémentaires : --set '{"n_ens": 4, "device": "mps"}' (JSON).

L'OOF et le test sont écrits dans realmlp/out/<name>_{oof,test}.npy ; l'évaluation (stacking) se fait dans un
processus séparé (realmlp/evaluate.py) pour ne pas mêler PyTorch et les autres bibliothèques.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import cv_loop, fold_rmse, load, rmse, save  # noqa: E402

N_THREADS = 2


def realmlp_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """fit_fn compatible cv_loop. Les poids d'échantillon ne sont pas supportés par pytabkit (ignorés).
    Le pli de validation sert à choisir la meilleure époque (checkpoint), comme l'early stopping des autres
    modèles. Catégories : dtype `category` -> one-hot (<= 9 modalités) ou embeddings (TD)."""
    import torch
    from pytabkit import RealMLP_TD_Regressor

    from models import _with_te

    torch.set_num_threads(N_THREADS)
    p = dict(p)
    seeds = p.pop("seeds", 1)
    if p.pop("te", False):
        Xtr, Xva, Xte = _with_te(Xtr, ytr, Xva, Xte)
    # NaN interdits dans les numériques : médiane du pli d'entraînement + indicatrices de manquants
    num = [c for c in Xtr.columns if not isinstance(Xtr[c].dtype, pd.CategoricalDtype)]
    nan_cols = [c for c in num if Xtr[c].isna().any()]
    med = Xtr[num].median()

    def prep(X):
        X = X.copy()
        for c in nan_cols:
            X[f"na_{c}"] = X[c].isna().astype(np.float32)
        X[num] = X[num].astype(float).fillna(med)
        return X

    Xtr, Xva, Xte = prep(Xtr), prep(Xva), prep(Xte)
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        t0 = time.time()
        m = RealMLP_TD_Regressor(random_state=42 + s, n_threads=N_THREADS, **p)
        m.fit(Xtr, np.asarray(ytr, float), X_val=Xva, y_val=np.asarray(yva, float))
        pva = pva + m.predict(Xva) / seeds
        pte = pte + m.predict(Xte) / seeds
        print(f"    seed {s} : {(time.time() - t0) / 60:.1f} min", flush=True)
    return np.asarray(pva).ravel(), np.asarray(pte).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--target", default="A")
    ap.add_argument("--te", type=int, default=1)
    ap.add_argument("--folds", default="all", help="'all' ou liste '0,1'")
    ap.add_argument("--set", default="{}", help="paramètres pytabkit en JSON")
    ap.add_argument("--save", default=None, help="nom pour common.save (5 plis seulement)")
    a = ap.parse_args()

    params = {"te": bool(a.te), "device": "cpu", **json.loads(a.set)}
    only = None if a.folds == "all" else [int(k) for k in a.folds.split(",")]
    print(f"[{a.name}] cible {a.target}, params {params}, plis {only or 'tous'}", flush=True)
    t0 = time.time()
    oof, test = cv_loop(realmlp_fn, params, target=a.target, only_folds=only)
    _, _, _, _, folds, y = load()
    mask = ~np.isnan(oof)
    print(f"[{a.name}] RMSE {rmse(y[mask], oof[mask]):.3f}  temps {(time.time() - t0) / 60:.1f} min", flush=True)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    np.save(out / f"{a.name}_oof.npy", oof)
    np.save(out / f"{a.name}_test.npy", test)
    if only is None:
        print(f"[{a.name}] plis {fold_rmse(oof)}", flush=True)
        if a.save:
            save(a.save, oof, test)
            print(f"[{a.name}] sauvé dans results/{a.save}_*.npy", flush=True)


if __name__ == "__main__":
    main()
