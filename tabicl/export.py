"""Exporte les entrées de TabICLv2 (journal, étape 26) vers tabicl/data/, à envoyer sur un dataset Kaggle privé.

    uv run python tabicl/export.py

- base.npz : features v2 sans les catégorielles à forte cardinalité (portées par le target encoding et les
  fréquences), catégorielles restantes en codes entiers (-1 = manquant) ; lignes = train puis test.
- te10.npy : (10, n_train + n_test, 7) target encoding résiduel appris dans chacun des 10 sous-plis emboîtés
  (`cv.make_subfolds`, comme `main.py train --folds 10`), sur la cible C comme `models._with_te` : lignes
  d'entraînement encodées par K-fold interne, validation et test par le sous-pli d'entraînement entier.
- split10.npy : sous-pli de chaque ligne de train ; meta.npz : y (kg), yC (points DOTS), dots_den, plis.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv import make_subfolds, prepare  # noqa: E402
from features import CAT_COLS, TE_COLS, ResidualTargetEncoder  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data"
HIGH_CARD = ["WeightClassKg", "Country", "State", "Federation", "MeetCountry", "MeetState"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test, _, F_tr, F_te, folds = prepare()
    y = train["TotalKg"].to_numpy(float)
    F = pd.concat([F_tr, F_te], ignore_index=True)
    base = F.drop(columns=HIGH_CARD)
    for c in CAT_COLS:
        if c in base:
            base[c] = base[c].cat.codes.astype("int16")
    np.savez(OUT / "base.npz", X=base.to_numpy(np.float32), cols=np.array(base.columns),
             cats=np.array([c for c in CAT_COLS if c in base]))

    split = make_subfolds(train, folds, 2)
    yC = y * 500 / F_tr["dots_den"].to_numpy()
    n_tr = len(F_tr)
    te = None
    for j in range(10):
        tr, va = np.where(split != j)[0], np.where(split == j)[0]
        enc = ResidualTargetEncoder(TE_COLS)
        enc_tr = enc.fit_transform(F_tr.iloc[tr].reset_index(drop=True), yC[tr]).to_numpy()
        if te is None:  # une colonne par groupe de TE_COLS (dont les composites)
            te = np.zeros((10, len(F), enc_tr.shape[1]), dtype=np.float32)
        te[j, tr] = enc_tr
        te[j, va] = enc.transform(F_tr.iloc[va].reset_index(drop=True)).to_numpy()
        te[j, n_tr:] = enc.transform(F_te.reset_index(drop=True)).to_numpy()
        print(f"  sous-pli {j} : target encoding fait", flush=True)
    np.save(OUT / "te10.npy", te)
    np.save(OUT / "split10.npy", split)
    np.savez(OUT / "meta.npz", y=y, yC=yC, dots_den=F["dots_den"].to_numpy(), folds=folds, n_tr=n_tr,
             te_names=np.array(enc.names))
    print(f"{base.shape[1]} colonnes + {te.shape[2]} TE -> {OUT}")


if __name__ == "__main__":
    main()
