import numpy as np, pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
D = "data/competitions/performances-en-powerlifting/"
tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv")
print(tr.shape, te.shape)
print("id range train", tr.id.min(), tr.id.max(), "test", te.id.min(), te.id.max())
allid = pd.concat([tr.id, te.id]); print("ids unique:", allid.is_unique, "n:", len(allid), "max+1:", allid.max()+1)

# ---- 1. Is id informative? autocorrelation of TotalKg along id, meet-key constancy along id
s = tr.sort_values("id").reset_index(drop=True)
MEET = ["Federation","MeetYear","MeetCountry","MeetState"]
key = s[MEET].astype(str).agg("|".join, axis=1)
print("adjacent-id same meet key:", (key.values[1:] == key.values[:-1]).mean())
print("random-pair same meet key:", (key.values == np.random.permutation(key.values)).mean())
print("corr(TotalKg, TotalKg next id):", np.corrcoef(s.TotalKg.values[1:], s.TotalKg.values[:-1])[0,1])
print("corr(TotalKg, random):", np.corrcoef(s.TotalKg.values, np.random.permutation(s.TotalKg.values))[0,1])
print("adjacent same Sex:", (s.Sex.values[1:]==s.Sex.values[:-1]).mean(), "same Fed:", (s.Federation.values[1:]==s.Federation.values[:-1]).mean())

# ---- 2. value counts
for c in ["Sex","Equipment","Tested","Sanctioned"]:
    print(c, tr[c].value_counts().to_dict())
print("WeightClassKg formats:", tr.WeightClassKg.str.replace(r"\d+(\.\d+)?","N",regex=True).value_counts().to_dict())
print("AgeClass:", sorted(tr.AgeClass.unique())); print("BirthYearClass:", sorted(tr.BirthYearClass.unique()))
print("MeetYear:", tr.MeetYear.min(), tr.MeetYear.max())

# ---- 3. meet key stats
df = pd.concat([tr.assign(src="tr"), te.assign(src="te")], ignore_index=True)
df["meet"] = df[MEET].astype(str).agg("|".join, axis=1)
g = df.groupby("meet")
print("n meet groups:", g.ngroups, "size quantiles:", g.size().quantile([.1,.25,.5,.75,.9,.99]).to_dict())
print("test rows whose meet key appears in train:", df[df.src=="te"].meet.isin(set(df[df.src=="tr"].meet)).mean())
for c in ["Tested","Sanctioned"]:
    print(f"meet groups with >1 value of {c}:", (g[c].nunique()>1).mean())
# refine with Tested
df["meet2"] = df.meet + "|" + df.Tested.astype(str)
print("n meet2 groups:", df.meet2.nunique())

# ---- 4. OOF residual structure of the best ensemble
oof = np.load("outputs/oof/ensemble_stack.npy")
print("oof shape", oof.shape)
tr["res"] = tr.TotalKg - oof[:len(tr)] if oof.ndim==1 else tr.TotalKg - oof[:,0]
print("RMSE oof:", np.sqrt((tr.res**2).mean()))
tr["meet"] = tr[MEET].astype(str).agg("|".join, axis=1)
# ICC-like: variance of per-group residual mean vs. expected under noise
def icc(col):
    gg = tr.groupby(col)["res"]
    m, n = gg.mean(), gg.size()
    sig2 = tr.res.var()
    # expected var of group means under no group effect: sig2/n ; observed var weighted
    obs = (m**2 * n).sum()/n.sum(); exp = (sig2/n * n).sum()/n.sum()
    return len(m), obs, exp
for col in ["meet","Federation","MeetState","Country","State","WeightClassKg","AgeClass"]:
    k, obs, exp = icc(col); print(f"residual group-mean var {col}: groups={k} observed={obs:.1f} expected-if-noise={exp:.1f}")
tr["meet_div"] = tr.meet + "|" + tr.Sex + "|" + tr.Equipment
tr["meet_div_wc"] = tr.meet_div + "|" + tr.WeightClassKg
for col in ["meet_div","meet_div_wc"]:
    k, obs, exp = icc(col); print(f"residual group-mean var {col}: groups={k} observed={obs:.1f} expected-if-noise={exp:.1f}")

# ---- 5. residual by segment
tr["abs_res"] = tr.res.abs()
print(tr.groupby("Equipment").res.agg(["mean","std","size"]))
print(tr.groupby("Tested").res.agg(["mean","std","size"]))
print(tr.groupby(pd.cut(tr.BodyweightKg,[0,50,60,70,80,90,100,110,125,140,200,400])).res.agg(["mean","std","size"]))
print(tr.groupby(pd.cut(tr.Age,[0,14,18,23,30,40,50,60,70,100])).res.agg(["mean","std","size"]))
tr["half"] = (tr.Age*2%2==1)
print(tr.groupby("half").res.agg(["mean","std","size"]))
