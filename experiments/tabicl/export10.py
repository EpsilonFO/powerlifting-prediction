"""Target encoding pour les 10 sous-plis emboîtés (cv.make_subfolds), comme `main.py train --folds 10`.
te10.npy : (10, n_train + n_test, 7) ; split10.npy : sous-pli de chaque ligne de train."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load  # noqa: E402
from cv import make_subfolds  # noqa: E402
from features import ResidualTargetEncoder, TE_COLS  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
train, test, F_tr, F_te, folds, y = load()
split = make_subfolds(train, folds, 2)
yC = y * 500 / F_tr["dots_den"].to_numpy()
n_tr = len(F_tr)
te = np.zeros((10, n_tr + len(F_te), 7), dtype=np.float32)
for j in range(10):
    tr, va = np.where(split != j)[0], np.where(split == j)[0]
    enc = ResidualTargetEncoder(TE_COLS)
    te[j, tr] = enc.fit_transform(F_tr.iloc[tr].reset_index(drop=True), yC[tr]).to_numpy()
    te[j, va] = enc.transform(F_tr.iloc[va].reset_index(drop=True)).to_numpy()
    te[j, n_tr:] = enc.transform(F_te.reset_index(drop=True)).to_numpy()
np.save(OUT / "te10.npy", te)
np.save(OUT / "split10.npy", split)
print(te.shape, np.bincount(split))
