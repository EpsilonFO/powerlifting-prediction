import numpy as np, pandas as pd
pd.set_option("display.width", 200)
D = "data/competitions/performances-en-powerlifting/"
tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv")
oof = np.load("outputs/oof/ensemble_stack.npy")
tr["res"] = tr.TotalKg - oof
sig2 = tr.res.var(); print("sigma2", sig2, "rmse", np.sqrt((tr.res**2).mean()))
for d in (tr, te):
    d["by"] = np.floor(d.MeetYear - d.Age).astype(int)
    d["bw"] = d.BodyweightKg

def pairs(A, B, key, tol_bw, same_meet=None, exclude_self=True):
    """pairs (i in A, j in B) matching exactly on key, |by diff|<=1, |bw diff|<=tol_bw."""
    out = []
    a = A[key+["by","bw","MeetYear","id"]].reset_index().rename(columns={"index":"i"})
    b = B[key+["by","bw","MeetYear","id"]].reset_index().rename(columns={"index":"j"})
    for sh in (-1,0,1):
        bb = b.copy(); bb["by"] = bb["by"]+sh
        m = a.merge(bb, on=key+["by"], suffixes=("_a","_b"))
        m = m[(m.bw_a-m.bw_b).abs() <= tol_bw]
        if exclude_self: m = m[m.id_a != m.id_b]
        if same_meet is True: m = m[m.MeetYear_a == m.MeetYear_b]
        if same_meet is False: m = m[m.MeetYear_a != m.MeetYear_b]
        out.append(m[["i","j"]])
    return pd.concat(out).drop_duplicates()

BASE = ["Sex","Equipment","Country","State","Tested"]
configs = {
 "A fed+meetstate, same year, bw<=0.5":   (BASE+["Federation","MeetState","MeetCountry"], 0.5, True),
 "B fed+meetstate, same year, bw<=2.5":   (BASE+["Federation","MeetState","MeetCountry"], 2.5, True),
 "C fed, other year, bw<=2.5":           (BASE+["Federation"], 2.5, False),
 "D fed, other year, bw<=5":             (BASE+["Federation"], 5.0, False),
 "E no fed, other year, bw<=2.5":        (BASE, 2.5, False),
 "F parentfed, any year, bw<=2.5":       (BASE+["ParentFederation"], 2.5, None),
 "G no fed, any year, bw<=1":            (BASE, 1.0, None),
}
res = tr.res.to_numpy(); y = tr.TotalKg.to_numpy()
for name,(key,tol,sm) in configs.items():
    p = pairs(tr, tr, key, tol, sm)
    if len(p)==0: print(name, "no pairs"); continue
    r_i, r_j = res[p.i], res[p.j]
    rho = np.corrcoef(r_i, r_j)[0,1]
    # per-row: mean residual of matches
    g = p.groupby("i").j.agg(list)
    mres = g.apply(lambda js: res[js].mean()); cnt = g.apply(len)
    idx = mres.index.to_numpy(); m = mres.to_numpy(); c = cnt.to_numpy()
    # fit lambda on res_i ~ lambda * m (no intercept), compute MSE reduction on matched rows
    lam = (res[idx]*m).sum()/(m*m).sum()
    mse0 = (res[idx]**2).mean(); mse1 = ((res[idx]-lam*m)**2).mean()
    # shrinkage by count: fit lambda per count bucket
    gain_total = (mse0-mse1)*len(idx)/len(tr)
    # test coverage
    pt = pairs(te, tr, key, tol, sm, exclude_self=False)
    cov = pt.i.nunique()/len(te)
    print(f"{name:40s} pairs={len(p):7d} rows={len(idx):6d} ({len(idx)/len(tr):.1%}) rho={rho:.3f} lam={lam:.2f} "
          f"MSE matched {mse0:.0f}->{mse1:.0f}  RMSE overall {np.sqrt(sig2):.2f}->{np.sqrt(sig2-gain_total):.2f}  test cov={cov:.1%} "
          f"std|y_i-y_j|={np.std(y[p.i]-y[p.j]):.1f}")
