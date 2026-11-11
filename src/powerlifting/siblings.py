"""Correction « candidats même athlète » (journal, étapes 28 et 29).

Il n'y a pas d'identifiant d'athlète, mais un même athlète peut apparaître sur plusieurs lignes. Pour chaque ligne
a, on cherche des candidats b : même sexe, année de naissance compatible, poids de corps plausible, et au moins un
point commun géographique ou de fédération. La plupart des candidats sont d'autres personnes ; un modèle au niveau
de la paire apprend, à partir de ce qui rapproche a et b, quelle part du résidu de b reporter sur a.

Trois étages, tous appris sans jamais utiliser la cible de la ligne corrigée :
1. `candidate_pairs` : paires (a, b) parmi train + test, doublons exacts exclus (traités par postprocess.py).
2. `fit_pair_model` : LightGBM qui prédit le coefficient w tel que résidu(a) ≈ w × z(b), où z(b) est le résidu
   relatif de b ramené à l'échelle de a. Pour prédire les lignes du pli k, le modèle n'a vu aucune paire dont a ou
   b est dans le pli k.
3. `row_features` puis `fit_row_model` : agrégats par ligne (somme des w × z, somme et maximum des w…) et petit
   LightGBM qui prédit le résidu de la ligne.
"""

import numpy as np
import pandas as pd

from powerlifting.config import ID, N_FOLDS, TARGET
from powerlifting.features import RAW_FEATURES

# (clé de blocage, taille maximale d'une cellule clé × année de naissance, restriction éventuelle)
BLOCKS = [
    (["Sex", "Country", "State"], 300, "state_known"),
    (["Sex", "Federation", "MeetCountry", "MeetState"], 300, None),
    (["Sex", "Country", "Federation"], 300, None),
    (["Sex", "Country"], 80, None),
    (["Sex", "Federation"], 80, None),
    (["Sex", "MeetCountry", "MeetState"], 80, None),
]
SAME = ["Country", "State", "Federation", "ParentFederation", "MeetCountry", "MeetState", "Equipment", "Tested",
        "WeightClassKg", "AgeClass", "BirthYearClass"]
# Très peu de signal par paire (la plupart sont de fausses paires) : modèle petit et très régularisé. Réglé sur le
# pli 0 (journal, étape 29) : au-delà de ~250 arbres, ou avec des arbres plus grands, il sur-apprend.
PAIR_PARAMS = dict(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=1000, subsample=0.7,
                   subsample_freq=1, colsample_bytree=0.5, reg_lambda=100.0, verbose=-1, n_jobs=8)
ROW_PARAMS = dict(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0, verbose=-1, n_jobs=8)


def stack_frame(train: pd.DataFrame, test: pd.DataFrame, folds: np.ndarray, oof: np.ndarray, pred: np.ndarray):
    """Train + test empilés : `i` (position), `fold` (−1 pour le test), `pred` (OOF ou test), `res` (train seul)."""
    al = pd.concat([train[[ID, *RAW_FEATURES]], test[[ID, *RAW_FEATURES]]], ignore_index=True)
    al["i"] = np.arange(len(al))
    al["fold"] = np.r_[folds, -np.ones(len(test), dtype=int)]
    al["pred"] = np.r_[oof, pred]
    al["res"] = np.r_[train[TARGET].to_numpy(float) - oof, np.full(len(test), np.nan)]
    al["rk"] = pd.util.hash_pandas_object(al[RAW_FEATURES], index=False).to_numpy()
    half = (al["Age"] % 1) == 0.5  # âge approximatif : année de naissance connue exactement
    al["by_hi"] = np.where(half, al["MeetYear"] - al["Age"] - 0.5, al["MeetYear"] - al["Age"])
    al["by_lo"] = np.where(half, al["by_hi"], al["by_hi"] - 1)  # âge exact : deux années possibles
    al["half"] = half.astype(np.float32)
    return al


def _bw_ok(dbw, dy):
    """Écart de poids plausible pour un même athlète : 5 kg, plus 2 kg par an (5 ans au plus)."""
    return dbw <= 5 + 2 * np.minimum(dy, 5)


def candidate_pairs(al: pd.DataFrame) -> pd.DataFrame:
    """Paires orientées (i_a, i_b) de candidats, toutes lignes confondues (train et test)."""
    lo = al[["i", "by_lo"]].rename(columns={"by_lo": "by"})
    hi = al.loc[al["by_hi"] != al["by_lo"], ["i", "by_hi"]].rename(columns={"by_hi": "by"})
    E = pd.concat([lo, hi], ignore_index=True)
    n = len(al)
    bw, yr, rk = al["BodyweightKg"].to_numpy(), al["MeetYear"].to_numpy(), al["rk"].to_numpy()
    codes = []
    for key, cap, restrict in BLOCKS:
        cell = al.groupby(key, sort=False).ngroup().to_numpy()
        ok = np.ones(n, bool)
        for c in key:  # une modalité « Unknown » n'est pas un point commun
            ok &= (al[c] != "Unknown").to_numpy()
        e = E[ok[E["i"].to_numpy()]].copy()
        e["cell"] = cell[e["i"].to_numpy()]
        size = e.groupby(["cell", "by"])["i"].transform("size")
        e = e[(size >= 2) & (size <= cap)]
        m = e.merge(e, on=["cell", "by"], suffixes=("_a", "_b"))
        a, b = m["i_a"].to_numpy(), m["i_b"].to_numpy()
        keep = (a != b) & (rk[a] != rk[b]) & _bw_ok(np.abs(bw[a] - bw[b]), np.abs(yr[a] - yr[b]))
        codes.append(np.unique(a[keep].astype(np.int64) * n + b[keep]))
        print(f"  bloc {'×'.join(key):45s} {len(codes[-1]):>9d} paires", flush=True)
    code = np.unique(np.concatenate(codes))
    return pd.DataFrame({"i_a": (code // n).astype(np.int32), "i_b": (code % n).astype(np.int32)})


def pair_table(al: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Variables de chaque paire (a, b de référence = ligne du train d'un autre pli que a), et comptages par ligne
    calculés sur toutes les paires (train et test) : le nombre de candidats mesure la rareté du profil."""
    a, b = pairs["i_a"].to_numpy(), pairs["i_b"].to_numpy()
    col = {c: al[c].to_numpy() for c in al.columns}
    P = pd.DataFrame({"i_a": a, "i_b": b})
    P["dbw"] = np.abs(col["BodyweightKg"][a] - col["BodyweightKg"][b]).astype(np.float32)
    P["sdbw"] = (col["BodyweightKg"][a] - col["BodyweightKg"][b]).astype(np.float32)
    P["dy"] = np.abs(col["MeetYear"][a] - col["MeetYear"][b]).astype(np.float32)
    P["sdy"] = (col["MeetYear"][a] - col["MeetYear"][b]).astype(np.float32)
    P["ov"] = (np.minimum(col["by_hi"][a], col["by_hi"][b]) - np.maximum(col["by_lo"][a], col["by_lo"][b]) + 1).astype(np.float32)
    P["half_a"], P["half_b"] = col["half"][a], col["half"][b]
    for c in SAME:  # 1 : identiques et connus ; 0 : différents ; −1 : au moins un inconnu
        va, vb = col[c][a], col[c][b]
        P[f"s_{c}"] = np.where((va == "Unknown") | (vb == "Unknown"), -1, (va == vb).astype(int)).astype(np.float32)
    # comptages par ligne a, sur toutes les paires
    n = len(al)
    cnt = lambda mask: np.bincount(a[mask], minlength=n).astype(np.float32)
    counts = {
        "n_all": cnt(np.ones(len(a), bool)),
        "n_state": cnt(P["s_State"].to_numpy() == 1),
        "n_fed": cnt(P["s_Federation"].to_numpy() == 1),
        "n_ms": cnt(P["s_MeetState"].to_numpy() == 1),
        "n_ctry": cnt(P["s_Country"].to_numpy() == 1),
        "n_tight": cnt((P["dbw"].to_numpy() <= 1 + 0.5 * P["dy"].to_numpy())),
        "n_stfed": cnt((P["s_State"].to_numpy() == 1) & (P["s_Federation"].to_numpy() == 1)),
    }
    for k, v in counts.items():
        P[f"{k}_a"] = v[a]
    P["n_all_b"] = counts["n_all"][b]
    P["n_tight_b"] = counts["n_tight"][b]
    P["age_a"] = col["Age"][a].astype(np.float32)
    P["bw_a"] = col["BodyweightKg"][a].astype(np.float32)
    P["year_a"] = col["MeetYear"][a].astype(np.float32)
    P["male_a"] = (col["Sex"][a] == "M").astype(np.float32)
    P["pred_a"] = col["pred"][a].astype(np.float32)
    P["pred_ratio"] = (col["pred"][a] / np.maximum(col["pred"][b], 1.0)).astype(np.float32)
    P["fold_a"], P["fold_b"] = col["fold"][a].astype(np.int8), col["fold"][b].astype(np.int8)
    P = P[(P["fold_b"] >= 0) & (P["fold_b"] != P["fold_a"])].reset_index(drop=True)  # références : train, autre pli
    a, b = P["i_a"].to_numpy(), P["i_b"].to_numpy()
    rel = col["res"][b] / np.maximum(col["pred"][b], 50.0)  # résidu relatif de b
    P["z"] = (rel * col["pred"][a]).astype(np.float32)  # … ramené à l'échelle de a
    P["absrel_b"] = np.abs(rel).astype(np.float32)
    P["res_a"] = col["res"][a].astype(np.float32)
    P["rank_dbw"] = P.groupby("i_a")["dbw"].rank(method="first").astype(np.float32)
    P["n_ref_a"] = P.groupby("i_a")["i_b"].transform("size").astype(np.float32)
    return P


NOT_FEATURES = ["i_a", "i_b", "fold_a", "fold_b", "z", "res_a"]


def fit_pair_model(P: pd.DataFrame, params: dict | None = None) -> np.ndarray:
    """Coefficient w de chaque paire. Minimiser (res_a − w·z)² revient à régresser res_a / z avec les poids z².
    Le modèle qui prédit les paires dont a est dans le pli k (ou dans le test) n'a vu aucune ligne du pli k."""
    import lightgbm as lgb

    feats = [c for c in P.columns if c not in NOT_FEATURES]
    z = P["z"].to_numpy(float)
    zc = np.where(np.abs(z) < 1.0, np.where(z < 0, -1.0, 1.0), z)
    t = np.clip(P["res_a"].to_numpy(float) / zc, -50, 50)
    wgt = zc**2 / np.mean(zc**2)
    fa, fb = P["fold_a"].to_numpy(), P["fold_b"].to_numpy()
    w = np.zeros(len(P))
    X = P[feats]
    for k in range(N_FOLDS):
        tr = (fa >= 0) & (fa != k) & (fb != k)
        mdl = lgb.LGBMRegressor(**(params or PAIR_PARAMS), random_state=k).fit(X[tr], t[tr], sample_weight=wgt[tr])
        va = fa == k
        w[va] = mdl.predict(X[va])
        te = fa == -1
        w[te] += mdl.predict(X[te]) / N_FOLDS
        r = P["res_a"].to_numpy(float)[va]
        print(f"  pli {k} : {tr.sum()} paires d'apprentissage ; sur les paires du pli, "
              f"MSE {np.mean(r**2):.1f} -> {np.mean((r - w[va] * z[va])**2):.1f}", flush=True)
    return w


def row_features(al: pd.DataFrame, P: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    """Agrégats par ligne (train puis test) des coefficients et des résidus reportés."""
    d = pd.DataFrame({"i_a": P["i_a"].to_numpy(), "w": w, "wz": w * P["z"].to_numpy(float), "z": P["z"].to_numpy(float)})
    d["w_pos"] = d["w"].clip(lower=0)
    g = d.groupby("i_a")
    R = pd.DataFrame(index=np.arange(len(al)))
    R["sib_sum"] = g["wz"].sum()
    R["sib_w_sum"] = g["w_pos"].sum()
    R["sib_w_max"] = g["w"].max()
    R["sib_n_ref"] = g.size()
    R["sib_wmean"] = R["sib_sum"] / (1.0 + R["sib_w_sum"])  # reports multiples : moyenne rétrécie plutôt que somme
    top = d.sort_values("w", ascending=False).drop_duplicates("i_a").set_index("i_a")
    R["sib_top_z"], R["sib_top_wz"] = top["z"], top["wz"]
    first = P.drop_duplicates("i_a").set_index("i_a")
    for c in ["n_all_a", "n_state_a", "n_fed_a", "n_ms_a", "n_ctry_a", "n_tight_a", "n_stfed_a"]:
        R[c] = first[c]
    R = R.fillna(0.0)
    R["Age"], R["BodyweightKg"], R["MeetYear"] = al["Age"].to_numpy(), al["BodyweightKg"].to_numpy(), al["MeetYear"].to_numpy()
    R["male"] = (al["Sex"] == "M").astype(int).to_numpy()
    return R


def fit_row_model(R: pd.DataFrame, al: pd.DataFrame, seeds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Correction par ligne : OOF sur les plis de référence pour le train, modèle appris sur tout le train pour le
    test. Retourne (correction train, correction test)."""
    import lightgbm as lgb

    fold = al["fold"].to_numpy()
    is_tr = fold >= 0
    X, y = R[is_tr], al.loc[is_tr, "res"].to_numpy(float)
    f = fold[is_tr]
    oof = np.zeros(is_tr.sum())
    for k in range(N_FOLDS):
        va = f == k
        for s in range(seeds):
            oof[va] += lgb.LGBMRegressor(**ROW_PARAMS, random_state=s).fit(X[~va], y[~va]).predict(X[va]) / seeds
    te = np.zeros((~is_tr).sum())
    for s in range(seeds):
        te += lgb.LGBMRegressor(**ROW_PARAMS, random_state=s).fit(X, y).predict(R[~is_tr]) / seeds
    return oof, te


def sibling_experiment(name: str, train: pd.DataFrame, test: pd.DataFrame, folds: np.ndarray) -> dict:
    """Applique la correction à l'expérience `name` et la journalise sous `{name}_pairs` (OOF honnête : la
    correction d'une ligne du pli k ne dépend d'aucune cible du pli k pour les deux premiers étages, et vient d'un
    modèle par ligne appris sur les autres plis)."""
    from powerlifting.config import OOF_DIR, PRED_DIR
    from powerlifting.cv import rmse
    from powerlifting.experiment_log import log_experiment, make_result, read_log

    y = train[TARGET].to_numpy(float)
    oof, pred = np.load(OOF_DIR / f"{name}.npy"), np.load(PRED_DIR / f"{name}.npy")
    al = stack_frame(train, test, folds, oof, pred)
    P = pair_table(al, candidate_pairs(al))
    n_tr = P.loc[P["fold_a"] >= 0, "i_a"].nunique()
    print(f"  [{name}] {len(P)} paires ; lignes avec candidat : {n_tr / len(train):.1%} du train, "
          f"{P.loc[P['fold_a'] < 0, 'i_a'].nunique() / len(test):.1%} du test")
    c_tr, c_te = fit_row_model(row_features(al, P, fit_pair_model(P)), al)
    new = f"{name}_pairs"
    np.save(OOF_DIR / f"{new}.npy", oof + c_tr)
    np.save(PRED_DIR / f"{new}.npy", pred + c_te)
    src = read_log().set_index("name").loc[name]
    scores = [rmse(y[folds == k], (oof + c_tr)[folds == k]) for k in range(N_FOLDS)]
    result = make_result(new, src["model"], src["target"], src["outliers"], src["seeds"], src["n_features"], scores, 0.0)
    log_experiment(result)
    print(f"  [{new}] correction : écart-type {c_tr.std():.2f} kg (train), {c_te.std():.2f} kg (test)")
    print(f"  [{new}] CV RMSE = {result['cv_rmse']:.3f} (avant : {src['cv_rmse']:.3f})  plis {result['fold_rmse']}")
    return result
