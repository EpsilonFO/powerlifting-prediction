"""TabICLv2 sur les plis de référence du projet powerlifting (GPU Kaggle).
FOLDS / TARGETS / SUBSAMPLE se règlent ci-dessous. Sorties : /kaggle/working/tabicl_{target}_fold{k}.npz (pred_va, pred_te en kg)."""
import glob, json, os, subprocess, sys, time
_root = os.path.dirname(glob.glob("/kaggle/input/**/base.npz", recursive=True)[0])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                *glob.glob(f"{_root}/*.whl")], check=True)  # pas d'internet : paquets fournis dans le dataset
import numpy as np, torch
from tabicl import TabICLRegressor

FOLDS = [0]
TARGETS = ["C", "A"]
N_EST = 8

root = os.path.dirname(glob.glob("/kaggle/input/**/base.npz", recursive=True)[0])
d = np.load(f"{root}/base.npz", allow_pickle=True); te = np.load(f"{root}/te.npy"); m = np.load(f"{root}/meta.npz", allow_pickle=True)
folds, y, yC, den, n_tr = m["folds"], m["y"], m["yC"], m["dots_den"], int(m["n_tr"])
print(torch.cuda.get_device_name(0), torch.cuda.device_count(), flush=True)
log = []
for k in FOLDS:
    X = np.hstack([d["X"], te[k]]).astype(np.float32)
    Xtr_all, Xte = X[:n_tr], X[n_tr:]
    tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
    for tgt in TARGETS:
        yt = yC if tgt == "C" else y
        scale_va = den[va] / 500 if tgt == "C" else 1.0
        scale_te = den[n_tr:] / 500 if tgt == "C" else 1.0
        t0 = time.time()
        r = TabICLRegressor(n_estimators=N_EST, device="cuda", random_state=0, verbose=False,
                            model_path=f"{root}/tabicl-regressor-v2-20260212.ckpt", allow_auto_download=False)
        r.fit(Xtr_all[tr], yt[tr]); t_fit = time.time() - t0
        p_va = r.predict(Xtr_all[va]) * scale_va; t_va = time.time() - t0
        score = float(np.sqrt(np.mean((p_va - y[va]) ** 2)))
        print(f"fold {k} target {tgt}: RMSE {score:.3f}  fit {t_fit:.0f}s  +val {t_va:.0f}s", flush=True)
        p_te = r.predict(Xte) * scale_te; t_all = time.time() - t0
        print(f"   +test {t_all:.0f}s  peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        np.savez(f"/kaggle/working/tabicl_{tgt}_fold{k}.npz", pred_va=p_va, pred_te=p_te, va=va)
        log.append({"fold": k, "target": tgt, "rmse": score, "sec": round(t_all)})
        json.dump(log, open("/kaggle/working/log.json", "w"))
        del r; torch.cuda.empty_cache()
print(log)
