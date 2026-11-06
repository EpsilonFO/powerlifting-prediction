"""Ablation TabPrep sur LightGBM tuné (cible C, 1 seed). Usage :

    uv run python tabprep/run.py VARIANTE [VARIANTE ...] [--folds 0]

Chaque variante ajoute ses colonnes TabPrep (calculées dans le pli) aux features v2 + TE résiduel du projet.
Résultats ajoutés à tabprep/results.jsonl ; OOF / test des runs 5 plis dans tabprep/cache/.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import cv_loop, fold_rmse, load, rmse  # noqa: E402
import models  # noqa: E402
from cv import load_params  # noqa: E402

import tp  # noqa: E402

N_JOBS = 3
SEED_OFFSET = 0
PARAMS = {**models.MODELS["lgbm"]["params"], **load_params("lgbm")}
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

MEET_KEYS = {"Meet": tp.MEET, "Meet_Div": tp.MEET + ["Sex", "Equipment"]}
GB_NUMS_F = ["Age", "BodyweightKg", "MeetYear", "birth_year", "bw_over_wc", "dots_den"]


def gens(name):
    """Liste de générateurs (fonctions (Xtr, ytr, Xva, Xte) -> ((A, B, C), info)) pour une variante."""
    G = {
        "base": [],
        # RSFC fidèle : 16 variables brutes, 50 sous-ensembles de taille 2..p-1, arrondi 2 déc., TE sans lissage
        "rsfc": [lambda *a: tp.rsfc(*a)],
        # RSFC adapté : petits sous-ensembles (2-4), numériques en 16 quantiles, TE lissé (20)
        "rsfc_small": [lambda *a: tp.rsfc(*a, max_size=4, bins=16, alpha=20.0, include_full=False)],
        "rsfc_small100": [lambda *a: tp.rsfc(*a, n_subsets=100, max_size=4, bins=16, alpha=20.0, include_full=False)],
        # GroupBy relatifs fidèles : 13 catégorielles brutes × 3 numériques brutes, ratio + pct_rank
        "gb": [lambda *a: tp.groupby_rel(*a)],
        # GroupBy + numériques dérivées + diff
        "gb_wide": [lambda *a: tp.groupby_rel(*a, cats=tp.RAW_CATS + ["Sex_Equip", "Equip_Tested"],
                                              nums=GB_NUMS_F, ops=("ratio", "diff", "pct_rank"))],
        # GroupBy relatifs à la compétition reconstituée (au-delà des rangs déjà testés : ratio / écart, plus de num.)
        "gb_wc": [lambda *a: tp.groupby_rel(*a, cats=["WeightClassKg"])],
        "gb_nowc": [lambda *a: tp.groupby_rel(*a, cats=[c for c in tp.RAW_CATS if c != "WeightClassKg"])],
        "gbT": [lambda *a: tp.groupby_rel(*a, fit_on="all")],
        "gb_diff": [lambda *a: tp.groupby_rel(*a, ops=("diff", "pct_rank"))],
        "gb_meetT": [lambda *a: tp.groupby_rel(*a, cats=[], nums=GB_NUMS_F, extra_keys=MEET_KEYS,
                                               ops=("ratio", "diff", "pct_rank"), fit_on="all")],
        "gb_meet": [lambda *a: tp.groupby_rel(*a, cats=[], nums=GB_NUMS_F, extra_keys=MEET_KEYS,
                                              ops=("ratio", "diff", "pct_rank"))],
        # OAFE fidèle sur les 3 numériques brutes (ordre 3)
        "oafe": [lambda *a: tp.oafe(*a, base=tp.RAW_NUMS)],
        # OAFE fidèle sur toutes les numériques des features v2 (budget 2000) ; variante 500
        "oafe_F": [lambda *a: tp.oafe(*a)],
        "oafe_F500": [lambda *a: tp.oafe(*a, max_new=500)],
        # CatPrep : TE (lissage 10) de chaque catégorielle brute + 100 combinaisons aléatoires d'ordre 2-3
        "cat": [lambda *a: tp.cat_combos(*a)],
    }
    out = []
    for part in name.split("+"):
        out += G[part]
    return out


# OAFE sur les colonnes de TE (appliqué après _with_te) : arithmétique entre niveaux de compétition / fédération...
POST = {
    "oafe_te": lambda Xtr, ytr, Xva, Xte: tp.oafe(
        Xtr, ytr, Xva, Xte, base=tp.RAW_NUMS + [c for c in Xtr.columns if c.startswith("te_")], max_new=500),
}


def make_fit(name, stats):
    parts = name.split("+")
    pre = "+".join(p for p in parts if p not in POST) or "base"
    post = [POST[p] for p in parts if p in POST]

    def fit(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
        import lightgbm as lgb

        t0 = time.time()
        new = [g(Xtr, ytr, Xva, Xte)[0] for g in gens(pre)]
        Xtr, Xva, Xte = models._with_te(Xtr, ytr, Xva, Xte)
        new += [g(Xtr, ytr, Xva, Xte)[0] for g in post]
        if new:
            Xtr = pd.concat([Xtr] + [n[0] for n in new], axis=1)
            Xva = pd.concat([Xva] + [n[1] for n in new], axis=1)
            Xte = pd.concat([Xte] + [n[2] for n in new], axis=1)
        t_feat = time.time() - t0
        p, seeds, n_est, patience = models._pop_common(p)
        m = lgb.LGBMRegressor(n_estimators=n_est, random_state=models.SEED + SEED_OFFSET, n_jobs=N_JOBS, verbose=-1, **p)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_X=Xva, eval_y=yva, eval_sample_weight=None if wva is None else [wva],
              callbacks=[lgb.early_stopping(patience, verbose=False)])
        imp = pd.Series(m.booster_.feature_importance("gain"), index=Xtr.columns)
        newcols = [c for n in new for c in n[0].columns]
        stats.append({"n_feats": Xtr.shape[1], "n_new": len(newcols), "t_feat": round(t_feat, 1),
                      "t_fit": round(time.time() - t0 - t_feat, 1),
                      "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2), "best_iter": int(m.best_iteration_),
                      "gain_share_new": round(float(imp[newcols].sum() / imp.sum()), 4) if newcols else 0.0,
                      "top_new": imp[newcols].sort_values(ascending=False).head(8).round(0).to_dict() if newcols else {}})
        return m.predict(Xva), m.predict(Xte)

    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--seed", type=int, default=0, help="décalage de la graine LightGBM (bruit de seed)")
    a = ap.parse_args()
    global SEED_OFFSET
    SEED_OFFSET = a.seed
    _, _, _, _, folds, y = load()
    for v0 in a.variants:
        v = v0 + (f"@s{a.seed}" if a.seed else "")
        stats = []
        t0 = time.time()
        print(f"=== {v} (plis {a.folds or 'tous'})", flush=True)
        oof, te = cv_loop(make_fit(v0, stats), PARAMS, target="C", only_folds=a.folds)
        ks = a.folds if a.folds else list(range(5))
        plis = [round(rmse(y[folds == k], oof[folds == k]), 3) for k in ks]
        res = {"variant": v, "folds": ks, "plis": plis, "minutes": round((time.time() - t0) / 60, 1),
               "rmse": round(rmse(y[np.isin(folds, ks)], oof[np.isin(folds, ks)]), 3), "stats": stats}
        if not a.folds:
            np.save(CACHE / f"{v}_oof.npy", oof)
            np.save(CACHE / f"{v}_test.npy", te)
        print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)
        with open(HERE / "results.jsonl", "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
