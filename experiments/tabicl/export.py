"""Exporte les entrées de TabICL pour les 5 plis de référence (exécution sur GPU Kaggle).

base.npz : features v2 sans les catégorielles à forte cardinalité (portées par le target encoding et les
fréquences), catégorielles restantes en codes entiers ; lignes = train puis test.
te.npy : (5, n_train + n_test, 7) target encoding appris dans chaque pli (cible C, comme models._with_te) :
lignes d'entraînement du pli encodées par K-fold interne, validation et test par le pli d'entraînement entier.
meta.npz : y (kg), yC (points DOTS), dots_den, folds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load  # noqa: E402
from features import CAT_COLS, ResidualTargetEncoder, TE_COLS  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
HIGH_CARD = ["WeightClassKg", "Country", "State", "Federation", "MeetCountry", "MeetState"]

train, test, F_tr, F_te, folds, y = load()
F = pd.concat([F_tr, F_te], ignore_index=True)
base = F.drop(columns=HIGH_CARD)
for c in CAT_COLS:
    if c in base:
        base[c] = base[c].cat.codes.astype("int16")  # -1 = manquant
np.savez(OUT / "base.npz", X=base.to_numpy(np.float32), cols=np.array(base.columns),
         cats=np.array([c for c in CAT_COLS if c in base]))

yC = y * 500 / F_tr["dots_den"].to_numpy()
n_tr = len(F_tr)
te_all = np.zeros((5, len(F), 7), dtype=np.float32)
for k in range(5):
    tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
    enc = ResidualTargetEncoder(TE_COLS)
    Xtr = F_tr.iloc[tr].reset_index(drop=True)
    te_all[k, tr] = enc.fit_transform(Xtr, yC[tr]).to_numpy()
    te_all[k, va] = enc.transform(F_tr.iloc[va].reset_index(drop=True)).to_numpy()
    te_all[k, n_tr:] = enc.transform(F_te.reset_index(drop=True)).to_numpy()
    names = enc.names
np.save(OUT / "te.npy", te_all)
np.savez(OUT / "meta.npz", y=y, yC=yC, dots_den=F["dots_den"].to_numpy(), folds=folds, n_tr=n_tr,
         te_names=np.array(names))
print(base.shape, te_all.shape, names)
