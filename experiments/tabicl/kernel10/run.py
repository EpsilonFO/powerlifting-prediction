"""TabICLv2, cible C, 10 sous-plis emboîtés, un sous-pli après l'autre sur un GPU T4.
Sorties : /kaggle/working/tabicl10_C_sub{j}.npz (pred_va, pred_te en kg, va)."""
import glob, json, os, subprocess, sys, time

root = os.path.dirname(glob.glob("/kaggle/input/**/base.npz", recursive=True)[0])
if len(sys.argv) == 1:  # processus parent : installe puis lance le worker
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                    *glob.glob(f"{root}/*.whl")], check=True)
    # un seul worker : deux en parallèle dépassent la RAM de la machine (tampon de sortie ~17 Go chacun)
    p = subprocess.run([sys.executable, __file__, "all"], env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    print("code de sortie :", p.returncode, flush=True)
    sys.exit(0)

import numpy as np, torch
from tabicl import TabICLRegressor
os.makedirs("/kaggle/tmp/offload", exist_ok=True)
d = np.load(f"{root}/base.npz", allow_pickle=True); te = np.load(f"{root}/te10.npy", mmap_mode="r")
m = np.load(f"{root}/meta.npz", allow_pickle=True); split = np.load(f"{root}/split10.npy")
y, yC, den, n_tr = m["y"], m["yC"], m["dots_den"], int(m["n_tr"])
for j in range(10):
    if os.path.exists(f"/kaggle/working/tabicl10_C_sub{j}.npz"):
        continue
    t0 = time.time()
    X = np.hstack([d["X"], te[j]]).astype(np.float32)
    tr, va = np.where(split != j)[0], np.where(split == j)[0]
    r = TabICLRegressor(n_estimators=8, device="cuda", random_state=0, verbose=False,
                        model_path=f"{root}/tabicl-regressor-v2-20260212.ckpt", allow_auto_download=False,
                        disk_offload_dir="/kaggle/tmp/offload")
    r.fit(X[tr], yC[tr])
    p_va = r.predict(X[va]) * den[va] / 500
    p_te = r.predict(X[n_tr:]) * den[n_tr:] / 500
    np.savez(f"/kaggle/working/tabicl10_C_sub{j}.npz", pred_va=p_va, pred_te=p_te, va=va)
    print(f"sous-pli {j} : RMSE {np.sqrt(np.mean((p_va - y[va]) ** 2)):.3f}  {time.time() - t0:.0f}s", flush=True)
    del r; torch.cuda.empty_cache()
