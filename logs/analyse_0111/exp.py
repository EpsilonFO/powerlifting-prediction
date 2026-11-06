"""Ablations rapides (1 seed, cible C, 5 plis, rien n'est sauvegardé) pour l'avis sur le prétraitement."""
import functools, sys, time
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du projet
from cv import prepare, run_cv, load_params
import models
from features import ResidualTargetEncoder, TE_COLS, TE_COMPOSITES, MEET, CAT_COLS

train, test, sample, F_tr, F_te, folds = prepare()
lgbm_p, mlp_p, cat_p = load_params("lgbm"), load_params("mlp"), load_params("cat")
ORIG_TE = models.ResidualTargetEncoder

def run(name, fn, params, Ftr=F_tr, Fte=F_te, base=None):
    # hérite des paramètres par défaut du modèle de base (learning_rate, subsample_freq...) comme main.py train
    models.MODELS[name] = {"fn": fn, "params": dict(models.MODELS.get(base, {}).get("params", {}))}
    t0 = time.time()
    r = run_cv(name, name, train, test, Ftr, Fte, folds, target="C", seeds=1, params=dict(params), save=False, verbose=False)
    print(f"RESULT {name:28s} {r['cv_rmse']:.3f}  {r.get('fold_rmse')}  ({(time.time()-t0)/60:.1f} min)", flush=True)

def lgbm_core(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """lgbm_fn sans le target encoding (déjà fait par l'appelant)."""
    import lightgbm as lgb
    p, seeds, n_est, patience = models._pop_common(p)
    m = lgb.LGBMRegressor(n_estimators=n_est, random_state=models.SEED, n_jobs=-1, verbose=-1, **p)
    m.fit(Xtr, ytr, sample_weight=wtr, eval_X=Xva, eval_y=yva, eval_sample_weight=None if wva is None else [wva],
          callbacks=[lgb.early_stopping(patience, verbose=False)])
    return m.predict(Xva), m.predict(Xte)

def with_encoder(enc_factory, core=lgbm_core):
    def fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
        models.ResidualTargetEncoder = enc_factory
        try:
            Xtr, Xva, Xte = models._with_te(Xtr, ytr, Xva, Xte)
        finally:
            models.ResidualTargetEncoder = ORIG_TE
        drop = [c for c in Xtr.columns if c.startswith("_")]
        return core(Xtr.drop(columns=drop), ytr, wtr, Xva.drop(columns=drop), yva, wva, Xte.drop(columns=drop), p)
    return fn

class TwoStageTE(ResidualTargetEncoder):
    """Résidu par rapport à une prédiction de base (colonne _base) au lieu de la moyenne Sex × Equipment."""
    def _stats(self, F, y):
        resid = np.asarray(y, float) - F["_base"].to_numpy(float)
        stats = {}
        for c in self.cols:
            codes, k = self._codes(F, c), len(F[c].cat.categories) + 1
            stats[c] = (np.bincount(codes, weights=resid, minlength=k), np.bincount(codes, minlength=k))
        for name, cols in self.composites.items():
            stats[name] = pd.Series(resid).groupby(self._keys(F, cols)).agg(["sum", "size"])
        return stats

which = sys.argv[1:] or ["lgbm", "mlp", "cat"]

if "lgbm" in which:
    run("lgbm_base", with_encoder(ORIG_TE), lgbm_p, base="lgbm")
    for s in (5, 50):
        run(f"lgbm_smooth{s}", with_encoder(functools.partial(ORIG_TE, smoothing=s)), lgbm_p, base="lgbm")
    comp = {**TE_COMPOSITES, "FedYear": ["Federation", "MeetYear"], "Fed_SexEq": ["Federation", "Sex", "Equipment"],
            "Meet_Div_WC": MEET + ["Sex", "Equipment", "WeightClassKg"]}
    run("lgbm_more_composites", with_encoder(functools.partial(ORIG_TE, composites=comp)), lgbm_p, base="lgbm")
    run("lgbm_meet_tested", with_encoder(functools.partial(ORIG_TE, composites={**TE_COMPOSITES, "Meet_Tested": MEET + ["Tested"]})), lgbm_p, base="lgbm")
    hgb = np.load("outputs/oof/hgb_raw.npy")
    F2tr, F2te = F_tr.copy(), F_te.copy()
    F2tr["_base"] = hgb * 500 / F_tr["dots_den"].to_numpy(); F2te["_base"] = 0.0
    run("lgbm_twostage_te", with_encoder(TwoStageTE), lgbm_p, F2tr, F2te, base="lgbm")

if "mlp" in which:
    def mlp_core(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
        return models.mlp_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p)
    run("mlp_base", models.mlp_fn, mlp_p)
    run("mlp_with_te", with_encoder(ORIG_TE, core=mlp_core), mlp_p)

if "cat" in which:
    def add_meet(F):
        F = F.copy()
        meet = F[MEET].astype(object).fillna("NA").astype(str).agg("|".join, axis=1)
        F["Meet"] = pd.Categorical(meet); F["Meet_Div"] = pd.Categorical(meet + "|" + F["Sex"].astype(object).fillna("NA").astype(str) + "|" + F["Equipment"].astype(object).fillna("NA").astype(str))
        return F
    def cat_meet(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
        models.CAT_COLS = CAT_COLS + ["Meet", "Meet_Div"]
        try:
            return models.cat_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p)
        finally:
            models.CAT_COLS = CAT_COLS
    run("cat_meet_cats", cat_meet, cat_p, add_meet(F_tr), add_meet(F_te), base="cat")
