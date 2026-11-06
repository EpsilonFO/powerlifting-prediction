"""Outils partagés pour évaluer une méthode nouvelle SANS toucher au projet principal.

Usage (depuis le dossier experiments/) :

    uv run python -c "..."            # l'environnement experiments/ a tabm, pytabkit, tabicl en plus
    from common import load, evaluate, save

    train, test, F_tr, F_te, folds, y = load()
    ... entraîner sur les 5 plis de référence (folds == k) ...
    save("tabm", oof, test_pred)      # -> experiments/results/tabm_{oof,test}.npy
    print(evaluate(oof))              # RMSE seule, plis, corrélation des résidus, gain dans le stacking à 33

Règles : ne rien écrire dans outputs/ ni modifier les fichiers .py du projet. Tout ce qui dépend de la cible
(target encoding…) doit être appris dans le pli d'entraînement.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import N_FOLDS, OOF_DIR, TARGET  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
ENSEMBLE_FILE = ROOT / "logs" / "ensemble_k10_v2_models.txt"
REF_MODELS = ["lgbm_C_k10", "mlp_C_plr_te_k10", "cat_C_meet_k10", "xgb_C_k10"]

_cache: dict = {}


def load():
    """train, test, F_tr, F_te (features v2 du projet), folds (5 plis de référence), y (TotalKg)."""
    if not _cache:
        from cv import prepare

        train, test, _, F_tr, F_te, folds = prepare()
        _cache.update(train=train, test=test, F_tr=F_tr, F_te=F_te, folds=folds,
                      y=train[TARGET].to_numpy(float))
    c = _cache
    return c["train"], c["test"], c["F_tr"], c["F_te"], c["folds"], c["y"]


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def fold_rmse(oof) -> list[float]:
    _, _, _, _, folds, y = load()
    return [round(rmse(y[folds == k], oof[folds == k]), 3) for k in range(N_FOLDS)]


def save(name: str, oof: np.ndarray, test_pred: np.ndarray) -> None:
    np.save(RESULTS / f"{name}_oof.npy", oof)
    np.save(RESULTS / f"{name}_test.npy", test_pred)


def cv_loop(fit_fn, params: dict | None = None, *, target: str = "C", only_folds=None, verbose=True):
    """Même boucle que `cv.run_cv` mais renvoie (oof, test) en kg sans rien écrire. `fit_fn` a la signature
    des modèles du projet : fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, params) -> (pred_va, pred_te) dans l'espace
    de la cible transformée (cible C = points DOTS, poids w = (dénominateur/500)², cf. features.TargetTransform).
    Exemple : `from models import mlp_fn; cv_loop(mlp_fn, {**mlp_json, "te": True})`."""
    import time

    from features import TargetTransform

    _, _, F_tr, F_te, folds, y = load()
    tt = TargetTransform(target)
    oof, pred_te = np.full(len(y), np.nan), np.zeros(len(F_te))
    ks = range(N_FOLDS) if only_folds is None else only_folds
    t0 = time.time()
    for k in ks:
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        Xtr, Xva = F_tr.iloc[tr].reset_index(drop=True), F_tr.iloc[va].reset_index(drop=True)
        p_va, p_te = fit_fn(Xtr, tt.forward(y[tr], Xtr), tt.weight(Xtr), Xva, tt.forward(y[va], Xva),
                            tt.weight(Xva), F_te, dict(params or {}))
        oof[va] = tt.inverse(p_va, Xva)
        pred_te += tt.inverse(p_te, F_te) / len(ks)
        if verbose:
            print(f"  pli {k} : RMSE {rmse(y[va], oof[va]):.3f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return oof, pred_te


def ensemble_members() -> list[str]:
    return [l.strip() for l in ENSEMBLE_FILE.read_text().splitlines() if l.strip()]


def evaluate(oof: np.ndarray, extra: list[np.ndarray] | None = None) -> dict:
    """Mesures standard d'un nouveau modèle (OOF en kg, sur les 5 plis de référence) :
    - rmse / plis : score seul ;
    - corr_resid : corrélation de ses résidus avec ceux des 4 meilleurs modèles de l'ensemble (plus elle est
      basse, plus il peut apporter de diversité) ;
    - stack_base / stack_new / delta : stacking Ridge (α = 10, + Sex / Equipment, validation imbriquée sur
      les 5 plis, exactement `ensemble._honest_stack`) des 33 modèles actuels, sans puis avec ce modèle ;
      delta_plis : écart par pli (négatif = gain). Référence attendue : stack_base ≈ 76,369.
    Attention : les 33 modèles sont entraînés à 10 plis (90 % du train) ; un modèle neuf entraîné à 5 plis est
    désavantagé d'environ 0,15-0,25 seul (journal, étape 23) : son gain dans le stacking est une borne basse."""
    from ensemble import _honest_stack, _meta_features

    _, _, F_tr, _, folds, y = load()
    assert oof.shape == y.shape and not np.isnan(oof).any(), "OOF incomplète"
    names = ensemble_members()
    O = np.column_stack([np.load(OOF_DIR / f"{n}.npy") for n in names])
    new = np.column_stack([oof] + list(extra or []))
    if "stack_base" not in _cache:
        _cache["stack_base"] = _honest_stack(_meta_features(O, F_tr), y, folds)
    base = _cache["stack_base"]
    stacked = _honest_stack(_meta_features(np.hstack([O, new]), F_tr), y, folds)
    r = y - oof
    corr = {m: round(float(np.corrcoef(r, y - np.load(OOF_DIR / f"{m}.npy"))[0, 1]), 3) for m in REF_MODELS}
    return {
        "rmse": round(rmse(y, oof), 3), "plis": fold_rmse(oof), "corr_resid": corr,
        "stack_base": round(rmse(y, base), 4), "stack_new": round(rmse(y, stacked), 4),
        "delta": round(rmse(y, stacked) - rmse(y, base), 4),
        "delta_plis": [round(rmse(y[folds == k], stacked[folds == k]) - rmse(y[folds == k], base[folds == k]), 4)
                       for k in range(N_FOLDS)],
    }
