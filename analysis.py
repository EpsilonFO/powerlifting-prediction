"""Phase 7 : importance des features (gain, permutation, SHAP), dépendances partielles, résidus par segment."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIG_DIR, OOF_DIR, OUT_DIR, SEED, TARGET  # noqa: E402
from cv import prepare, rmse  # noqa: E402
from features import ResidualTargetEncoder, build_features  # noqa: E402
from models import MODELS  # noqa: E402


def residual_tables(train: pd.DataFrame, oof: np.ndarray) -> None:
    """RMSE et biais de l'OOF par Sex, Equipment, Federation, tranche de poids de corps et décennie."""
    d = train.assign(
        err=train[TARGET] - oof,
        bw_bin=pd.cut(train["BodyweightKg"], [0, 60, 75, 90, 105, 125, 150, 400]).astype(str),
        decade=(train["MeetYear"] // 10 * 10).astype(str),
    )
    top_fed = d["Federation"].value_counts().index[:10]
    d["Federation_top"] = d["Federation"].where(d["Federation"].isin(top_fed), "(autres)")
    for col in ["Sex", "Equipment", "Tested", "bw_bin", "decade", "Federation_top"]:
        g = d.groupby(col)["err"].agg(n="size", rmse=lambda e: np.sqrt(np.mean(e**2)), biais="mean").round(2)
        print(f"\nRésidus par {col} :\n{g.sort_values('n', ascending=False).to_string()}")


def _fit_lgbm_fold0(train, F_tr, folds, y):
    import lightgbm as lgb
    from cv import load_params

    tr, va = np.where(folds != 0)[0], np.where(folds == 0)[0]
    Xtr, Xva = F_tr.iloc[tr].reset_index(drop=True), F_tr.iloc[va].reset_index(drop=True)
    te = ResidualTargetEncoder()
    Xtr, Xva_te = pd.concat([Xtr, te.fit_transform(Xtr, y[tr])], axis=1), pd.concat([Xva, te.transform(Xva)], axis=1)
    params = {**MODELS["lgbm"]["params"], **load_params("lgbm"), "learning_rate": 0.08}
    m = lgb.LGBMRegressor(n_estimators=3000, random_state=SEED, n_jobs=-1, verbose=-1, **params)
    m.fit(Xtr, y[tr], eval_X=Xva_te, eval_y=y[va], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, va, Xva_te


def feature_importance(m, Xva: pd.DataFrame, yva: np.ndarray, n_perm_rows: int = 8000, n_shap_rows: int = 3000):
    rng = np.random.default_rng(SEED)
    base = rmse(yva, m.predict(Xva))
    idx = rng.choice(len(Xva), min(n_perm_rows, len(Xva)), replace=False)
    sub, ysub = Xva.iloc[idx].reset_index(drop=True), yva[idx]
    ref = rmse(ysub, m.predict(sub))
    perm = {}
    for c in Xva.columns:
        loss = []
        for _ in range(2):
            s = sub.copy()
            s[c] = s[c].iloc[rng.permutation(len(s))].array  # .array conserve le dtype category
            loss.append(rmse(ysub, m.predict(s)) - ref)
        perm[c] = np.mean(loss)
    contrib = m.booster_.predict(Xva.iloc[idx[:n_shap_rows]], pred_contrib=True)[:, :-1]
    imp = pd.DataFrame({
        "gain": pd.Series(m.booster_.feature_importance("gain"), index=m.booster_.feature_name()),
        "permutation_rmse_increase": pd.Series(perm),
        "mean_abs_shap": pd.Series(np.abs(contrib).mean(0), index=Xva.columns),
    })
    imp["gain"] = imp["gain"] / imp["gain"].sum()
    print(f"\nRMSE fold 0 du modèle d'analyse : {base:.3f}")
    print(imp.sort_values("mean_abs_shap", ascending=False).head(20).round(4).to_string())
    imp.to_csv(OUT_DIR / "feature_importance.csv")

    top = imp.sort_values("mean_abs_shap").tail(20)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    ax[0].barh(top.index, top["mean_abs_shap"])
    ax[0].set_title("SHAP moyen |kg|")
    ax[1].barh(top.index, top["permutation_rmse_increase"])
    ax[1].set_title("Permutation : hausse de RMSE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)
    return imp


def partial_dependence(m, train, test, F_tr, folds, va, Xva_te, n_rows=1500, n_grid=15):
    """Dépendance partielle de BodyweightKg, Age, MeetYear. On modifie la colonne *brute* puis on reconstruit
    toutes les features dérivées (log, Wilks, âge², ...), sinon elles resteraient incohérentes."""
    rng = np.random.default_rng(SEED)
    pos = rng.choice(len(va), n_rows, replace=False)  # positions dans le pli de validation
    rows = va[pos]  # indices correspondants dans train
    te_cols = [c for c in Xva_te.columns if c.startswith("te_")]
    te_rows = Xva_te.iloc[pos][te_cols].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["BodyweightKg", "Age", "MeetYear"]):
        grid = np.unique(np.quantile(train[col], np.linspace(0.02, 0.98, n_grid)))
        curve = []
        for v in grid:
            tr_mod = train.copy()
            tr_mod.loc[train.index[rows], col] = v
            Fm, _ = build_features(tr_mod, test)
            X = pd.concat([Fm.iloc[rows].reset_index(drop=True), te_rows], axis=1)[Xva_te.columns]
            curve.append(m.predict(X).mean())
        ax.plot(grid, curve, marker="o")
        ax.set_title(f"Dépendance partielle : {col}")
        ax.set_ylabel("TotalKg prédit (kg)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "partial_dependence.png", dpi=130)
    plt.close(fig)


def run_analysis(oof_name: str = "lgbm_A") -> None:
    train, test, _, F_tr, _, folds = prepare()
    y = train[TARGET].to_numpy(float)
    oof = np.load(OOF_DIR / f"{oof_name}.npy")
    print(f"RMSE OOF de {oof_name} : {rmse(y, oof):.3f}")
    residual_tables(train, oof)
    m, va, Xva_te = _fit_lgbm_fold0(train, F_tr, folds, y)
    feature_importance(m, Xva_te, y[va])
    partial_dependence(m, train, test, F_tr, folds, va, Xva_te)
    print(f"\nFigures et tableaux dans {OUT_DIR}")
