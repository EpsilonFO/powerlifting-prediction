"""Réimplémentation légère des générateurs TabPrep (Tschalzev, Erickson et al., 2026, arXiv 2606.02384) d'après le
code d'AutoGluon (`autogluon.features.generators` : `rsfc.py`, `groupby.py`, `arithmetic/`, `oof_target_encoder.py`,
`CategoricalInteractionFeatureGenerator`). AutoGluon n'est PAS installé : on reprend les mêmes règles en numpy/pandas.

Chaque générateur prend (Xtr, ytr, Xva, Xte) et renvoie trois DataFrames de nouvelles colonnes numériques. Tout ce
qui utilise la cible n'est appris que sur Xtr (TE hors pli interne à 5 plis pour les lignes de Xtr, statistiques sur
tout Xtr pour Xva / Xte). Les statistiques non supervisées (moyennes de groupe, rangs) sont aussi apprises sur Xtr
seul, comme dans TabPrep.
"""

from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

RAW_NUMS = ["Age", "BodyweightKg", "MeetYear"]
RAW_CATS = ["Sex", "Equipment", "AgeClass", "BirthYearClass", "WeightClassKg", "Tested", "Country", "State",
            "Federation", "ParentFederation", "MeetCountry", "MeetState", "Sanctioned"]
MEET = ["Federation", "MeetYear", "MeetCountry", "MeetState"]


# ----------------------------------------------------------------------------- utilitaires

def _codes(s: pd.Series) -> np.ndarray:
    """Codes entiers (NaN -> une modalité à part, comme le hachage de pandas)."""
    c, _ = pd.factorize(s.astype(object) if isinstance(s.dtype, pd.CategoricalDtype) else s, use_na_sentinel=False)
    return c.astype(np.int64)


def combine_keys(cols_codes: list[np.ndarray]) -> np.ndarray:
    """Clé jointe de plusieurs colonnes codées (refactorisée à chaque étape : pas de débordement)."""
    key = cols_codes[0]
    for c in cols_codes[1:]:
        key, _ = pd.factorize(key * (int(c.max()) + 1) + c)
        key = key.astype(np.int64)
    return key


def oof_te(key: np.ndarray, ntr: int, y: np.ndarray, alpha: float, n_inner: int = 5, seed: int = 42) -> np.ndarray:
    """OOF target encoding façon `OOFTargetEncodingFeatureGenerator` (régression) : clé sur toutes les lignes
    (train du pli puis val puis test), cible des ntr premières. Lignes train : encodées par K-fold interne ;
    autres lignes : statistiques de tout le train. Lissage vers la moyenne NON pondérée des moyennes de modalités
    (comme AutoGluon) ; modalité absente -> cette moyenne."""
    key, _ = pd.factorize(key)
    k = int(key.max()) + 1
    out = np.empty(len(key))

    def enc(idx_fit, idx_apply):
        kc = key[idx_fit]
        cnt = np.bincount(kc, minlength=k).astype(float)
        s = np.bincount(kc, weights=y[idx_fit], minlength=k)
        seen = cnt > 0
        mean = np.where(seen, s / np.maximum(cnt, 1), np.nan)
        prior = np.nanmean(mean)
        e = np.where(seen, (s + alpha * prior) / (cnt + alpha), prior)
        return e[key[idx_apply]]

    tr = np.arange(ntr)
    for a, b in KFold(n_inner, shuffle=True, random_state=seed).split(tr):
        out[b] = enc(a, b)
    out[ntr:] = enc(tr, np.arange(ntr, len(key)))
    return out


def _split(arr: np.ndarray, ntr: int, nva: int, names: list[str], idx):
    df = pd.DataFrame(arr, columns=names)
    a, b, c = df.iloc[:ntr], df.iloc[ntr:ntr + nva], df.iloc[ntr + nva:]
    return (a.set_index(idx[0]), b.set_index(idx[1]), c.set_index(idx[2]))


def _concat(Xtr, Xva, Xte, cols):
    return pd.concat([Xtr[cols], Xva[cols], Xte[cols]], ignore_index=True)


# ----------------------------------------------------------------------------- RSFC

def rsfc(Xtr, ytr, Xva, Xte, *, base=None, n_subsets=50, min_size=2, max_size=None, round_num=2, bins=None,
         alpha=0.0, include_full=True, seed=42):
    """Random Subset Feature Compression. Défauts = AutoGluon : 50 sous-ensembles (le premier = toutes les
    variables de base), tailles tirées uniformément dans [2, p-1], numériques arrondies à 2 décimales,
    OOF-TE sans lissage (alpha = 0). `bins` : discrétisation par quantiles (variante, au lieu de l'arrondi)."""
    base = list(RAW_NUMS + RAW_CATS if base is None else base)
    D = _concat(Xtr, Xva, Xte, base)
    ntr, nva = len(Xtr), len(Xva)
    codes = {}
    for c in base:
        s = D[c]
        if not isinstance(s.dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(s):
            if bins:
                edges = np.unique(np.nanquantile(s.iloc[:ntr], np.linspace(0, 1, bins + 1)[1:-1]))
                s = pd.Series(np.digitize(s, edges))
            elif round_num is not None:
                s = s.round(round_num)
        codes[c] = _codes(s)
    # ordre AutoGluon : catégorielles par cardinalité croissante, puis binaires, entiers, flottants
    cats = sorted([c for c in base if isinstance(D[c].dtype, pd.CategoricalDtype)], key=lambda c: D[c].nunique())
    nums = sorted([c for c in base if c not in cats], key=lambda c: D[c].nunique())
    feats = np.array(cats + nums, dtype=object)
    p = len(feats)
    rng = np.random.default_rng(seed)
    kmax = min(max_size if max_size is not None else p - 1, p - 1)
    subsets, seen = ([tuple(sorted(feats))] if include_full else []), set()
    for _ in range(50 * n_subsets):
        if len(subsets) >= n_subsets:
            break
        k = int(rng.integers(min_size, kmax + 1))
        sub = tuple(sorted(rng.choice(feats, size=k, replace=False).tolist()))
        if sub in seen:
            continue
        seen.add(sub)
        subsets.append(sub)
    out = np.column_stack([oof_te(combine_keys([codes[c] for c in sub]), ntr, ytr, alpha, seed=seed)
                           for sub in subsets])
    names = [f"RSFC_{i}" for i in range(len(subsets))]
    return _split(out, ntr, nva, names, (Xtr.index, Xva.index, Xte.index)), subsets


# ----------------------------------------------------------------------------- GroupBy relatifs

def _rank_cats_small_counts(Xtr, cats, min_count=20, top_k=10):
    scores = {}
    for c in cats:
        cnt = Xtr[c].value_counts(dropna=True)
        cnt = cnt[cnt >= min_count].sort_values()
        if len(cnt) < 2:
            v = np.full(top_k, -np.inf)
        else:
            v = cnt.to_numpy(float)[:top_k]
            v = np.pad(v, (0, top_k - v.size), constant_values=np.inf)
        scores[c] = tuple(v)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def groupby_rel(Xtr, ytr, Xva, Xte, *, cats=None, nums=None, extra_keys=None, max_pairs=500, ops=("ratio", "pct_rank"),
                min_num_card=10, fit_on="train"):
    """GroupBy relatifs (défaut TabPrep : pct_rank + ratio à la moyenne du groupe ; `diff` en option). Statistiques
    apprises sur Xtr. Paires : catégorielles classées par taille de leurs plus petits groupes (≥ 20), numériques
    par cardinalité décroissante, budget `max_pairs`. `extra_keys` : {nom: [colonnes]} groupes composites
    (ex. compétition), ajoutés aux catégorielles. fit_on="all" : statistiques (sans la cible) calculées sur
    train + val + test (transductif, comme `meet_size` du projet) au lieu du seul train du pli."""
    cats = list(RAW_CATS if cats is None else cats)
    nums = [c for c in (RAW_NUMS if nums is None else nums) if Xtr[c].nunique() >= min_num_card]
    ntr, nva = len(Xtr), len(Xva)
    D = _concat(Xtr, Xva, Xte, list(dict.fromkeys(cats + nums + sum((extra_keys or {}).values(), []))))
    G = pd.DataFrame({c: _codes(D[c]) for c in cats})
    na = {c: D[c].isna().to_numpy() for c in cats}
    for name, cols in (extra_keys or {}).items():
        G[name] = combine_keys([_codes(D[c]) for c in cols])
        na[name] = np.zeros(len(D), bool)
    Gtr = G.iloc[:ntr]
    ranked = _rank_cats_small_counts(Gtr.where(~pd.DataFrame(na).iloc[:ntr]), list(G.columns))
    ranked_nums = sorted(nums, key=lambda c: -Xtr[c].nunique())
    pairs = [(c, n) for c in ranked for n in ranked_nums][:max_pairs]
    cols, names = [], []
    for c, n in pairs:
        g = G[c].to_numpy()
        x = D[n].to_numpy(float)
        k = int(g.max()) + 1
        nfit = ntr if fit_on == "train" else len(g)
        gtr, xtr = g[:nfit], x[:nfit]
        ok = ~np.isnan(xtr)
        cnt = np.bincount(gtr[ok], minlength=k)
        mean = np.bincount(gtr[ok], weights=xtr[ok], minlength=k) / np.maximum(cnt, 1)
        m = np.where((cnt[g] > 0) & ~na[c], mean[g], np.nan)
        if "ratio" in ops:
            cols.append(x / (m + 1e-8)); names.append(f"{n}__ratio__{c}")
        if "diff" in ops:
            cols.append(x - m); names.append(f"{n}__minus__{c}")
        if "pct_rank" in ops:
            # rang dans la distribution du groupe sur le train : #(train du groupe <= x) / taille du groupe
            order = np.lexsort((xtr[ok], gtr[ok]))
            gs, xs = gtr[ok][order], xtr[ok][order]
            start = np.r_[0, np.cumsum(cnt)[:-1]]
            # position globale de x dans le bloc trié de son groupe : searchsorted sur (groupe, x)
            comb_sorted = gs.astype(float) * 1e6 + xs  # x < 1e6 (âge, poids, année, tailles)
            pos = np.clip(np.searchsorted(comb_sorted, g.astype(float) * 1e6 + x, side="right") - start[g], 0, cnt[g])
            pr = np.where((cnt[g] > 0) & ~np.isnan(x) & ~na[c], pos / np.maximum(cnt[g], 1), 0.5)
            cols.append(pr); names.append(f"{n}__pctrank__{c}")
    out = np.column_stack(cols).astype(np.float32)
    return _split(out, ntr, nva, names, (Xtr.index, Xva.index, Xte.index)), pairs


# ----------------------------------------------------------------------------- OAFE

def _basic_filter(D: pd.DataFrame, min_card=3, nan_thr=0.95, mode_thr=0.9) -> list[str]:
    keep = []
    for c in D.columns:
        s = D[c]
        if s.isna().mean() > nan_thr or s.nunique() < min_card:
            continue
        if s.value_counts(normalize=True, dropna=False).iloc[0] > mode_thr:
            continue
        keep.append(c)
    return keep


def oafe(Xtr, ytr, Xva, Xte, *, base=None, max_order=3, max_new=2000, seed=42, ops=("/", "*", "-", "+")):
    """Ordered Arithmetic Feature Expansion, sélection « random » (défaut TabPrep) : toutes les paires (tirées dans
    un ordre aléatoire), opérateurs dans l'ordre /, *, −, +, puis b/a ; ordre 3 = (opération d'ordre 2) ∘ variable
    de base absente de l'opération ; budget `max_new`. Variables de base : numériques passant le filtre de base
    (cardinalité ≥ 3, mode ≤ 90 %, NaN ≤ 95 %), remises à l'échelle (division par l'écart-type du train) pour
    éviter les débordements, comme `rescale_avoid_overflow`."""
    if base is None:
        base = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c]) and not
                isinstance(Xtr[c].dtype, pd.CategoricalDtype)]
    base = _basic_filter(Xtr[base])
    ntr, nva = len(Xtr), len(Xva)
    D = _concat(Xtr, Xva, Xte, base).astype(float)
    sd = D.iloc[:ntr].std().replace(0, 1)
    V = {c: (D[c] / sd[c]).to_numpy(np.float32) for c in base}
    rng = np.random.default_rng(seed)
    f = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}
    feats, names = [], []
    pairs = list(combinations(base, 2))
    pairs = [pairs[i] for i in rng.choice(len(pairs), min(len(pairs), max_new), replace=False)]
    order2 = []
    with np.errstate(all="ignore"):
        for op in ops:
            for a, b in pairs:
                order2.append((f"({a}{op}{b})", f[op](V[a], V[b]), {a, b}))
        if "/" in ops:
            for a, b in pairs:
                order2.append((f"({b}/{a})", V[b] / V[a], {a, b}))
        for nm, v, _ in order2:
            feats.append(v); names.append(nm)
        if max_order >= 3 and len(feats) < max_new:
            rem = max_new - len(feats)
            cand = [(o, b) for o, b in product(range(len(order2)), base) if b not in order2[o][2]]
            cand = [cand[i] for i in rng.choice(len(cand), min(len(cand), rem), replace=False)]
            seen = set()
            for op in ("/", "-", "*", "+"):
                if op not in ops:
                    continue
                for o, b in cand:
                    nm = f"({order2[o][0]}{op}{b})"
                    if op in "*+":
                        key = (op, tuple(sorted((order2[o][0], b))))
                        if key in seen:
                            continue
                        seen.add(key)
                    feats.append(f[op](order2[o][1], V[b])); names.append(nm)
    feats, names = feats[:max_new], names[:max_new]
    out = np.column_stack(feats).astype(np.float32)
    out[~np.isfinite(out)] = np.nan
    return _split(out, ntr, nva, names, (Xtr.index, Xva.index, Xte.index)), base


# ----------------------------------------------------------------------------- CatPrep (combinaisons TE)

def cat_combos(Xtr, ytr, Xva, Xte, *, cats=None, n=100, max_order=3, alpha=10.0, seed=42, single=True):
    """OOF-TE (lissage 10, 5 plis) de chaque catégorielle et de `n` combinaisons aléatoires d'ordre 2 à 3 (défaut
    `CategoricalInteractionFeatureGenerator` + `OOFTargetEncodingFeatureGenerator`)."""
    cats = list(RAW_CATS if cats is None else cats)
    ntr, nva = len(Xtr), len(Xva)
    D = _concat(Xtr, Xva, Xte, cats)
    codes = {c: _codes(D[c]) for c in cats}
    rng = np.random.default_rng(seed)
    subs, seen = [], set()
    for _ in range(50 * n):
        if len(subs) >= n:
            break
        k = int(rng.integers(2, max_order + 1))
        s = tuple(sorted(rng.choice(cats, size=k, replace=False).tolist()))
        if s not in seen:
            seen.add(s); subs.append(s)
    groups = ([(c,) for c in cats] if single else []) + subs
    out = np.column_stack([oof_te(combine_keys([codes[c] for c in g]), ntr, ytr, alpha, seed=seed) for g in groups])
    names = ["cte_" + "|".join(g) for g in groups]
    return _split(out, ntr, nva, names, (Xtr.index, Xva.index, Xte.index)), groups
