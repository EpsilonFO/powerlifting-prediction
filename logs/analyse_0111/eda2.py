import numpy as np, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
D = "data/competitions/performances-en-powerlifting/"
tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv")
oof = np.load("outputs/oof/ensemble_stack.npy"); tr["pred"]=oof; tr["res"]=tr.TotalKg-oof
tr["ratio"]=tr.TotalKg/tr.BodyweightKg
# mixture check: ratio histogram by sex for adults 20-40 raw
a = tr[(tr.Age.between(20,40))&(tr.Equipment=="Raw")]
for s in "MF":
    r = a[a.Sex==s].ratio
    print(s, "ratio quantiles", r.quantile([.001,.01,.05,.1,.25,.5,.75,.9,.99]).round(2).to_dict())
    print(s, "hist", np.histogram(r, bins=np.arange(0,10,0.5))[0])
print("low totals (<150):"); print(tr[tr.TotalKg<150].sort_values("TotalKg").head(15)[["Age","BodyweightKg","Sex","Equipment","Federation","MeetCountry","MeetYear","AgeClass","TotalKg","pred"]])
print("BW>200:"); print(tr[tr.BodyweightKg>200][["Age","BodyweightKg","Sex","Equipment","WeightClassKg","Federation","MeetYear","TotalKg","pred"]])
# contribution of extreme residuals to MSE
r2 = tr.res**2; print("share of MSE from |res|>250:", r2[tr.res.abs()>250].sum()/r2.sum(), "n=", (tr.res.abs()>250).sum())
print("share of MSE from |res|>150:", r2[tr.res.abs()>150].sum()/r2.sum(), "n=", (tr.res.abs()>150).sum())
# lbs signature
lb = tr.BodyweightKg*2.20462
tr["lbs"] = ((lb - lb.round()).abs()<0.02)
print("lbs-like BW share:", tr.lbs.mean(), "by MeetCountry top:", tr.groupby("MeetCountry").lbs.mean().sort_values(ascending=False).head(5).round(2).to_dict())
MEET=["Federation","MeetYear","MeetCountry","MeetState"]
g=tr.groupby(MEET).lbs.agg(["mean","size"]); print("meet groups size>=10 with mixed lbs/kg (0.2<mean<0.8):", ((g["size"]>=10)&(g["mean"].between(0.2,0.8))).sum(), "of", (g["size"]>=10).sum())
# kids
print("Age<10:", (tr.Age<10).sum(), tr[tr.Age<10].res.abs().mean())
# residual vs number of decimals of BW
dec = tr.BodyweightKg.astype(str).str.split(".").str[1].str.rstrip("0").str.len().fillna(0)
print(tr.groupby(dec).res.agg(["mean","std","size"]))
# Mx and country stats
print(tr.groupby("Sex").res.agg(["mean","std","size"]))
# TE smoothing: distribution of meet group sizes in train only
k = tr[MEET].astype(str).agg("|".join,axis=1); print("train meet sizes:", k.value_counts().describe().round(1).to_dict())
