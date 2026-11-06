import time; t=time.time()
import sys; sys.path.insert(0, ".")
from common import load
from models import nn_arrays, _with_te
from features import TargetTransform
import numpy as np
train, test, F_tr, F_te, folds, y = load()
print(F_tr.shape, F_te.shape, time.time()-t)
import models
print("CAT_COLS", models.CAT_COLS)
tr, va = np.where(folds!=0)[0], np.where(folds==0)[0]
Xtr, Xva = F_tr.iloc[tr].reset_index(drop=True), F_tr.iloc[va].reset_index(drop=True)
tt = TargetTransform("C")
Xtr2, Xva2, Xte2 = _with_te(Xtr, tt.forward(y[tr], Xtr), Xva, F_te)
A, B, C, cards, cont = nn_arrays(Xtr2, Xva2, Xte2)
print(A.shape, "cards", cards, "n_cont", len(cont))
num_cols = [c for c in Xtr2.columns if c not in models.CAT_COLS]
print(num_cols)
print("cont", [num_cols[i] for i in cont])
print("nunique", {c: Xtr2[c].nunique() for c in num_cols})
print(time.time()-t)
