"""Méta-modèles non linéaires / contextuels pour l'ensemble à 37 modèles (journal, étape 26.1 : Ridge = 76,214).

Toutes les variantes sont évaluées en validation imbriquée sur les 5 plis de référence, comme `ensemble._honest_stack`
(méta-modèle appris sur 4 plis d'OOF, évalué sur le 5e). Usage (depuis experiments/) :

    PYTHONPATH=. uv run python meta/meta.py ridge fwls_seg lgbm_resid ...
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from common import ROOT, load, rmse
from config import N_FOLDS, OOF_DIR, PRED_DIR
from ensemble import _meta_features

NAMES = [l.strip() for l in (ROOT / "logs" / "ensemble_k10_v3_models.txt").read_text().splitlines() if l.strip()]
_, _, F, Fte, folds, y = load()
O = np.column_stack([np.load(OOF_DIR / f"{n}.npy") for n in NAMES])
T = np.column_stack([np.load(PRED_DIR / f"{n}.npy") for n in NAMES])
CTX_NUM = ["Age", "BodyweightKg", "MeetYear", "dots_den", "meet_size", "div_size", "bw_over_wc", "freq_Federation"]
CTX_CAT = ["Sex", "Equipment", "Tested", "Sex_Equip", "AgeClass"]


def ridge_stack(Ztr, ytr, Zva, alpha=10.0, w=None):
    return Ridge(alpha=alpha).fit(Ztr, ytr, sample_weight=w).predict(Zva)


def interactions(Om, Fm, kind, ref):
    """Feature-weighted linear stacking (Sill et al. 2009) : chaque prédiction multipliée par des variables de
    contexte -> poids des modèles qui dépendent du contexte, le méta-modèle reste linéaire."""
    base = _meta_features(Om, Fm)
    if kind == "seg":
        D = pd.get_dummies(Fm["Sex_Equip"].astype(str)).reindex(columns=ref["seg_cols"], fill_value=0).to_numpy(float)
        D = D[:, 1:]  # 1re modalité = référence
    else:  # continues standardisées (stats du train complet, sans cible) + indicatrice de manquant de l'âge
        X = Fm[ref["cont"]].astype(float)
        D = ((X.fillna(ref["med"]) - ref["mu"]) / ref["sd"]).clip(-3, 3).to_numpy()
        if kind == "both":
            S = pd.get_dummies(Fm["Sex_Equip"].astype(str)).reindex(columns=ref["seg_cols"], fill_value=0).to_numpy(float)
            D = np.hstack([D, S[:, 1:]])
    Oc = Om - ref["omean"]  # centré : les interactions restent lisibles et la Ridge mieux conditionnée
    inter = np.concatenate([Oc[:, :, None] * D[:, None, :]], axis=2).reshape(len(Om), -1)
    return np.hstack([base, D, inter])


def ref_stats(cont=("Age", "BodyweightKg")):
    cont = list(cont)
    X = F[cont].astype(float)
    med = X.median()
    Xf = X.fillna(med)
    return {"seg_cols": sorted(F["Sex_Equip"].astype(str).unique()), "cont": cont, "med": med,
            "mu": Xf.mean(), "sd": Xf.std(), "omean": O.mean(0)}


def run_fwls(kind, alpha, cont=("Age", "BodyweightKg")):
    ref = ref_stats(cont)
    Z = interactions(O, F, kind, ref)
    honest = np.zeros(len(y))
    for k in range(N_FOLDS):
        va = folds == k
        honest[va] = ridge_stack(Z[~va], y[~va], Z[va], alpha)
    return honest


def ctx_frame(Fm):
    X = Fm[CTX_NUM].astype(float).copy()
    for c in CTX_CAT:
        X[c] = Fm[c].astype("category")
    return X


def run_lgbm(mode, params=None, top=None):
    """mode = "resid" : LightGBM appris sur le résidu d'une Ridge honnête (Ridge imbriquée dans les 4 plis
    d'entraînement), entrées = prédiction Ridge + contexte (+ prédictions des `top` modèles en écart à la Ridge).
    mode = "direct" : LightGBM sur [prédictions, contexte] directement."""
    import lightgbm as lgb

    p = {"learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 300, "subsample": 0.8, "subsample_freq": 1,
         "colsample_bytree": 0.8, "reg_lambda": 10.0, "n_jobs": 6, "verbose": -1, "max_cat_to_onehot": 8,
         **(params or {})}
    Z = _meta_features(O, F)
    C = ctx_frame(F).reset_index(drop=True)
    honest = np.zeros(len(y))
    iters = []
    for k in range(N_FOLDS):
        tr_folds = [j for j in range(N_FOLDS) if j != k]
        va = folds == k
        tr = ~va
        # Ridge honnête sur les lignes d'entraînement (validation imbriquée à l'intérieur des 4 plis)
        r_tr = np.zeros(len(y))
        for j in tr_folds:
            m = folds == j
            r_tr[m] = ridge_stack(Z[tr & ~m], y[tr & ~m], Z[m])
        r_va = ridge_stack(Z[tr], y[tr], Z[va])
        r_all = r_tr.copy()
        r_all[va] = r_va
        X = C.copy()
        X["ridge"] = r_all
        if mode == "direct":
            for i, n in enumerate(NAMES):
                X[n] = O[:, i]
            target = y
        else:
            for i in (top or []):
                X[f"d_{NAMES[i]}"] = O[:, i] - r_all
            target = y - r_all
        # early stopping sur un des 4 plis d'entraînement, puis réapprentissage sur les 4 avec ce nombre d'arbres
        es = folds == tr_folds[0]
        fit_rows, es_rows = tr & ~es, es
        m = lgb.LGBMRegressor(n_estimators=3000, **p)
        m.fit(X[fit_rows], target[fit_rows], eval_set=[(X[es_rows], target[es_rows])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        n_it = max(int(m.best_iteration_ * 1.1), 10)
        iters.append(n_it)
        m = lgb.LGBMRegressor(n_estimators=n_it, **p).fit(X[tr], target[tr])
        pred = m.predict(X[va])
        honest[va] = pred if mode == "direct" else r_va + pred
    return honest, iters


def report(name, honest, base=None, extra=""):
    plis = [rmse(y[folds == k], honest[folds == k]) for k in range(N_FOLDS)]
    s = {"variant": name, "rmse": round(rmse(y, honest), 4), "plis": [round(v, 3) for v in plis]}
    if base is not None:
        s["delta"] = round(rmse(y, honest) - rmse(y, base), 4)
        s["delta_plis"] = [round(rmse(y[folds == k], honest[folds == k]) - rmse(y[folds == k], base[folds == k]), 4)
                           for k in range(N_FOLDS)]
    if extra:
        s["info"] = extra
    print(json.dumps(s, ensure_ascii=False), flush=True)
    with open(ROOT / "experiments" / "meta" / "results.jsonl", "a") as f:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    Z = _meta_features(O, F)
    base = np.zeros(len(y))
    for k in range(N_FOLDS):
        va = folds == k
        base[va] = ridge_stack(Z[~va], y[~va], Z[va])
    report("ridge (référence)", base)
    # 5 meilleurs modèles seuls, pour les écarts dans lgbm_resid
    top5 = list(np.argsort([rmse(y, O[:, i]) for i in range(len(NAMES))])[:5])
    for v in sys.argv[1:]:
        t0 = time.time()
        if v.startswith("fwls_"):  # fwls_seg_10, fwls_cont_10, fwls_both_100, fwls_cont4_100
            _, kind, a = v.split("_")
            cont = ("Age", "BodyweightKg")
            if kind == "cont4":
                kind, cont = "cont", ("Age", "BodyweightKg", "MeetYear", "meet_size")
            h, info = run_fwls(kind, float(a), cont), ""
        elif v == "lgbm_resid":
            h, it = run_lgbm("resid"); info = f"arbres {it}"
        elif v == "lgbm_resid_top":
            h, it = run_lgbm("resid", top=top5); info = f"arbres {it}"
        elif v == "lgbm_direct":
            h, it = run_lgbm("direct"); info = f"arbres {it}"
        else:
            raise SystemExit(f"variante inconnue : {v}")
        report(v, h, base, f"{info} {(time.time() - t0) / 60:.1f} min".strip())
        np.save(ROOT / "experiments" / "meta" / f"{v}_honest.npy", h)
