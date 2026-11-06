"""Réseaux : M7 MLP avec embeddings et PLR (journal, étape 13), TabM et RealMLP (étape 26). Les réseaux PyTorch
eux-mêmes sont dans `powerlifting.nets`."""

import numpy as np
import pandas as pd

from powerlifting.config import SEED
from powerlifting.features import CAT_COLS
from powerlifting.models.common import device, with_te


def nn_arrays(Xtr, Xva, Xte, min_count: int = 5):
    """Prétraitement des réseaux, ajusté sur le pli d'entraînement : une matrice float32 par jeu,
    `[codes des catégories | numériques écrêtées (0,5 %-99,5 %) et standardisées | indicatrices de manquants]`.
    Codes : 0 = modalité rare (< min_count) ou inconnue, 1 = manquant. Retourne aussi la cardinalité de chaque
    catégorie et les colonnes numériques continues (> 2 valeurs), candidates au plongement PLR."""
    num_cols = [c for c in Xtr.columns if c not in CAT_COLS]
    nan_cols = [c for c in num_cols if Xtr[c].isna().any()]
    med, lo, hi = Xtr[num_cols].median(), Xtr[num_cols].quantile(0.005), Xtr[num_cols].quantile(0.995)

    def clipped(X):
        return X[num_cols].astype(float).fillna(med).clip(lo, hi, axis=1)

    mu, sd = clipped(Xtr).mean(), clipped(Xtr).std().replace(0, 1)
    vocab = {}
    for c in CAT_COLS:
        vc = Xtr[c].value_counts()
        vocab[c] = {v: i + 2 for i, v in enumerate(vc.index[vc >= min_count])}

    def encode(X):
        codes = [X[c].astype(object).pipe(lambda s: s.map(vocab[c]).where(s.notna(), 1).fillna(0)) for c in CAT_COLS]
        num = ((clipped(X) - mu) / sd).to_numpy()
        return np.hstack([np.stack(codes, 1).astype(float), num, X[nan_cols].isna().to_numpy(float)]).astype(np.float32)

    cards = [len(vocab[c]) + 2 for c in CAT_COLS]
    cont_idx = [i for i, c in enumerate(num_cols) if Xtr[c].nunique() > 2]
    return encode(Xtr), encode(Xva), encode(Xte), cards, cont_idx


def mlp_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """Embeddings des catégories + MLP (briques dans `nets/mlp.py`). `plr` (dict k / d / sigma) active le plongement
    périodique des variables continues ; `history` (liste) reçoit les courbes d'apprentissage de la 1re seed."""
    import torch

    from powerlifting.nets.mlp import TabularInput, TabularMLP, fit, predict

    p = dict(p)
    seeds, hidden, drop = p.pop("seeds", 1), p.pop("hidden", (256, 128, 64)), p.pop("dropout", 0.15)
    plr, history, min_count = p.pop("plr", None), p.pop("history", None), p.pop("min_count", 5)
    torch.set_num_threads(8)

    if p.pop("te", False):  # target encoding en entrée numérique (journal, étape 25.5) : --set te=true
        Xtr, Xva, Xte = with_te(Xtr, ytr, Xva, Xte)
    Atr, Ava, Ate, cards, cont_idx = nn_arrays(Xtr, Xva, Xte, min_count)
    Atr, Ava, Ate = torch.from_numpy(Atr), torch.from_numpy(Ava), torch.from_numpy(Ate)
    n_num = Atr.shape[1] - len(cards)
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        torch.manual_seed(SEED + s)
        net = TabularMLP(TabularInput(cards, n_num, cont_idx, plr), hidden, drop)
        fit(net, Atr, ytr, wtr, Ava, yva, wva, history=history if s == 0 else None, **p)
        pva, pte = pva + predict(net, Ava) / seeds, pte + predict(net, Ate) / seeds
    return pva, pte


def tabm_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """TabM (BatchEnsemble de 32 têtes, plongements linéaires par morceaux), réseau dans `nets/tabm.py`.
    Défauts retenus dans `archive/experiments/tabm/REPORT.md` : catégories en embeddings, target encoding."""
    from powerlifting.nets.tabm import fit_tabm

    p = dict(p)
    seeds = p.pop("seeds", 1)
    p["device"] = device(p.get("device", "auto"))
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        a, b = fit_tabm(Xtr, ytr, wtr, Xva, yva, wva, Xte, {**p, "seed": SEED + s})
        pva, pte = pva + a / seeds, pte + b / seeds
    return pva, pte


def realmlp_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """RealMLP (Holzmüller et al., NeurIPS 2024), réglages par défaut « TD » de pytabkit. pytabkit n'accepte pas de
    poids d'échantillon : avec la cible C, la perte est la MSE en points DOTS (meilleure que A quand même, étape 26).
    Le pli de validation choisit la meilleure époque. NaN numériques : médiane du pli + indicatrices `na_*`."""
    import torch
    from pytabkit import RealMLP_TD_Regressor

    p = dict(p)
    seeds, threads = p.pop("seeds", 1), p.pop("threads", 2)
    p["device"] = device(p.get("device", "auto"))
    torch.set_num_threads(threads)
    if p.pop("te", True):
        Xtr, Xva, Xte = with_te(Xtr, ytr, Xva, Xte)
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
        m = RealMLP_TD_Regressor(random_state=SEED + s, n_threads=threads, **p)
        m.fit(Xtr, np.asarray(ytr, float), X_val=Xva, y_val=np.asarray(yva, float))
        pva = pva + np.asarray(m.predict(Xva)).ravel() / seeds
        pte = pte + np.asarray(m.predict(Xte)).ravel() / seeds
    return pva, pte
