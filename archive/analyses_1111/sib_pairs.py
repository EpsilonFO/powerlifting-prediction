import numpy as np, pandas as pd
pd.set_option("display.width", 250)
D = "data/competitions/performances-en-powerlifting/"
tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv")
tr["res"] = tr.TotalKg - np.load("outputs/oof/ensemble_stack.npy")
tr["fold"] = pd.read_csv("data/folds.csv").iloc[:, -1].to_numpy()
print(pd.read_csv("data/folds.csv").head(2))
half = (tr.Age % 1) == 0.5
print("x.5 share", half.mean(), "other frac parts", (tr.Age%1).value_counts().head(5).to_dict())
def expand(d):
    d = d.reset_index().rename(columns={"index": "i"})
    h = (d.Age % 1) == 0.5
    a = d[h].assign(by=(d.MeetYear - d.Age - 0.5)[h])
    b = d[~h].assign(by=(d.MeetYear - d.Age)[~h])
    c = d[~h].assign(by=(d.MeetYear - d.Age - 1)[~h])
    return pd.concat([a, b, c])
E = expand(tr)
cols = ["i","by","BodyweightKg","MeetYear","Federation","Equipment","res","TotalKg","Age","MeetState","State","Country"]
def study(key, name, mask=None):
    e = E if mask is None else E[mask(E)]
    c2=[c for c in cols if c not in key]; m = e[key+c2].merge(e[key+c2], on=key+["by"], suffixes=("_a","_b"))
    for c in key:
        m[c+"_a"]=m[c]; m[c+"_b"]=m[c]
    m = m[m.i_a != m.i_b].drop_duplicates(["i_a","i_b"])
    m["dbw"] = (m.BodyweightKg_a - m.BodyweightKg_b).abs()
    m["dy"] = (m.MeetYear_a - m.MeetYear_b).abs()
    m["ncand"] = m.groupby("i_a").i_b.transform("size")
    print(f"\n=== {name}: pairs={len(m)} rows={m.i_a.nunique()} ({m.i_a.nunique()/len(tr):.1%})")
    def rho(x): return pd.Series({"n": len(x), "rho_res": np.corrcoef(x.res_a, x.res_b)[0,1] if len(x)>30 else np.nan, "sd_dy": (x.TotalKg_a-x.TotalKg_b).std()})
    print(m.groupby(pd.cut(m.ncand,[0,1,2,4,8,16,64,1e9])).apply(rho))
    print(m.groupby(pd.cut(m.dbw,[-1,0.01,1,2.5,5,10,1e3])).apply(rho))
    print(m.groupby(pd.cut(m.dy,[-1,0,1,2,4,8,100])).apply(rho))
    t = m[(m.dbw<=5)]
    print("dbw<=5 by samefed:", t.groupby(t.Federation_a==t.Federation_b).apply(rho).to_dict("index"))
    t2 = t[(t.Federation_a==t.Federation_b)]
    t2 = t2.assign(nc=t2.groupby("i_a").i_b.transform("size"))
    print("dbw<=5 & same fed, by ncand:"); print(t2.groupby(pd.cut(t2.nc,[0,1,2,4,8,16,1e9])).apply(rho))
    return m
m1 = study(["Sex","Country","State"], "State known", lambda e: e.State!="Unknown")
m2 = study(["Sex","Country","State","Federation"], "State unknown, same fed", lambda e: e.State=="Unknown")
