import sys, warnings
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
from powerlifting import siblings as sb
from powerlifting.postprocess import row_keys, replace_duplicates
from powerlifting.data import load_data
from powerlifting.config import TARGET
S = sys.argv[1]
al, P = pd.read_pickle(S + "/P_v1.pkl"); d = pd.read_pickle(S + "/pairs_v2.pkl"); w = d["w"]; R0 = d["R"]
assert (d["P"].i_a.to_numpy() == P.i_a.to_numpy()).all()
train, test, _ = load_data(); y = train[TARGET].to_numpy(float); ntr = len(train); folds = al.fold.to_numpy()[:ntr]
oof = al.pred.to_numpy()[:ntr]; keys = row_keys(train)
def dd(p):
    p = p.copy()
    for k in range(5):
        va = folds == k; p[va], _ = replace_duplicates(p[va], keys[va], keys[~va], y[~va])
    return p
def ev(name, R, seeds=2):
    is_tr = al.fold.to_numpy() >= 0; X = R[is_tr]; res = y - oof; c = np.zeros(ntr)
    for k in range(5):
        va = folds == k
        for s in range(seeds): c[va] += lgb.LGBMRegressor(**sb.ROW_PARAMS, random_state=s).fit(X[~va], res[~va]).predict(X[va])/seeds
    q = dd(oof + c)
    print(f"{name:40s} {np.mean([np.sqrt(((y-q)[folds==k]**2).mean()) for k in range(5)]):.3f}", [round(float(np.sqrt(((y-q)[folds==k]**2).mean())),3) for k in range(5)], flush=True)
    return c
ev("ligne v2 (référence)", R0)
# --- accord entre candidats : top-10 par w
K = 10
D = pd.DataFrame({"i_a": P.i_a.to_numpy(), "w": np.clip(w, 0, None), "z": P.z.to_numpy(float), "dbw": P.dbw.to_numpy(), "dy": P.dy.to_numpy(),
                  "eq": P.s_Equipment.to_numpy(), "fed": P.s_Federation.to_numpy()})
D = D.sort_values(["i_a", "w"], ascending=[True, False]); D["r"] = D.groupby("i_a").cumcount()
T = D[D.r < K]
n = len(al); W = np.zeros((n, K)); Z = np.zeros((n, K))
W[T.i_a.to_numpy(), T.r.to_numpy()] = T.w.to_numpy(); Z[T.i_a.to_numpy(), T.r.to_numpy()] = T.z.to_numpy()
R1 = R0.copy()
for h in (15, 40):
    Kmat = np.exp(-(Z[:, :, None] - Z[:, None, :])**2 / (2*h*h)) * W[:, None, :]      # support de b par c
    sup = Kmat.sum(2) * (W > 0)
    best = sup.argmax(1); idx = np.arange(n)
    R1[f"mode_z_{h}"] = Z[idx, best]; R1[f"mode_sup_{h}"] = sup[idx, best]; R1[f"mode_others_{h}"] = sup[idx, best] - W[idx, best]
    kw = Kmat[idx, best, :]; R1[f"mode_mean_{h}"] = (kw * Z).sum(1) / np.maximum(kw.sum(1), 1e-9)
for j in range(3):
    R1[f"top{j}_w"] = W[:, j]; R1[f"top{j}_z"] = Z[:, j]
sw = np.maximum(W.sum(1), 1e-9); mz = (W*Z).sum(1)/sw
R1["wz_mean"] = mz; R1["wz_sd"] = np.sqrt((W*(Z-mz[:, None])**2).sum(1)/sw)
ev("+ accord entre candidats (mode)", R1)
# --- agrégats simples façon prototype
g = D.groupby("i_a"); R2 = R1.copy()
R2["z_mean"] = g.z.mean().reindex(R2.index); R2["z_sd"] = g.z.std().reindex(R2.index)
Dt = D[D.dbw <= 1 + 0.5*D.dy]; gt = Dt.groupby("i_a"); R2["zt_mean"] = gt.z.mean().reindex(R2.index); R2["zt_n"] = gt.size().reindex(R2.index).fillna(0)
De = D[(D.eq == 1) & (D.fed == 1)]; ge = De.groupby("i_a"); R2["ze_mean"] = ge.z.mean().reindex(R2.index); R2["ze_n"] = ge.size().reindex(R2.index).fillna(0)
nn = D.sort_values(["i_a", "dbw"]).drop_duplicates("i_a").set_index("i_a"); R2["nn_z"] = nn.z.reindex(R2.index); R2["nn_dbw"] = nn.dbw.reindex(R2.index); R2["nn_w"] = nn.w.reindex(R2.index)
c = ev("+ agrégats simples", R2)
pd.to_pickle(R2, S + "/R2.pkl")
