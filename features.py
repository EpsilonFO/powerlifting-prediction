"""Phases 1-3 du plan : nettoyage, feature engineering, transformations de cible, target encoding."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import SEED, TARGET

RAW_NUMS = ["Age", "BodyweightKg", "MeetYear"]
RAW_CATS = [
    "Sex", "Equipment", "AgeClass", "BirthYearClass", "WeightClassKg", "Tested", "Country",
    "State", "Federation", "ParentFederation", "MeetCountry", "MeetState", "Sanctioned",
]
RAW_FEATURES = RAW_NUMS + RAW_CATS

UNKNOWN_FLAG_COLS = ["State", "MeetState", "ParentFederation", "BirthYearClass"]
TE_COLS = ["Federation", "ParentFederation", "Country", "MeetCountry", "MeetState"]
# Target encoding de groupes définis par plusieurs colonnes : approximation d'une compétition (aucun identifiant
# de compétition dans les données) et de ses catégories. Les clés ne sont pas données telles quelles aux modèles.
MEET = ["Federation", "MeetYear", "MeetCountry", "MeetState"]
TE_COMPOSITES = {"Meet": MEET, "Meet_Div": MEET + ["Sex", "Equipment"]}
INTERACTIONS = {
    "Sex_Equip": ("Sex", "Equipment"),
    "Equip_Tested": ("Equipment", "Tested"),
}
# Fed_Equip (> 1 000 modalités) n'est plus une variable catégorielle : LightGBM sur-apprenait dessus (ablation du
# 20/10, -0,07). On garde seulement sa fréquence.
FREQ_INTERACTIONS = {"Fed_Equip": ("Federation", "Equipment")}
CAT_COLS = RAW_CATS + list(INTERACTIONS)
FREQ_COLS = TE_COLS + ["State", "Fed_Equip"]

# Coefficients (ordre croissant des puissances) des formules de points, en fonction du poids de corps.
_WILKS = {
    "M": [-216.0475144, 16.2606339, -0.002388645, -0.00113732, 7.01863e-06, -1.291e-08],
    "F": [594.31747775582, -27.23842536447, 0.82112226871, -0.00930733913, 4.731582e-05, -9.054e-08],
}
_DOTS = {
    "M": [-307.75076, 24.0900756, -0.1918759221, 0.0007391293, -0.000001093],
    "F": [-57.96288, 13.6175032, -0.1126655495, 0.0005158568, -0.0000010706],
}
# IPF GL : (A, B, C) avec dénominateur A - B * exp(-C * BW)
_IPF_GL = {
    ("M", "raw"): (1199.72839, 1025.18162, 0.00921),
    ("F", "raw"): (610.32796, 1045.59282, 0.03048),
    ("M", "equipped"): (1236.25115, 1449.21864, 0.01644),
    ("F", "equipped"): (758.63878, 949.31382, 0.02435),
}


def _poly(x: np.ndarray, coefs: list[float]) -> np.ndarray:
    return sum(c * x**i for i, c in enumerate(coefs))


def _class_bounds(s: pd.Series) -> pd.DataFrame:
    """'24-34' -> (24, 34, 29). Les bornes ouvertes ('70-999') sont plafonnées à 90."""
    b = s.str.extract(r"(\d+)-(\d+)").astype(float)
    lo, hi = b[0], b[1].clip(upper=90)
    return pd.DataFrame({"lo": lo, "hi": hi, "mid": (lo + hi) / 2})


# Features ajoutées par l'ablation du 20/10 (journal, section 18) ; absentes de la version v1.
V2_ONLY = ["birth_year", "meet_size", "div_size", "wc_size"]


def use_v1_features() -> None:
    """Revient aux features du 28/09 (v1) pour réentraîner les anciens modèles à l'identique : `Fed_Equip`
    redevient catégorielle et le target encoding de la compétition disparaît. Modifie en place CAT_COLS et
    TE_COMPOSITES, partagés avec models.py ; à appeler avant `build_features`."""
    if "Fed_Equip" not in CAT_COLS:
        CAT_COLS.append("Fed_Equip")
    TE_COMPOSITES.clear()


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features sans usage de la cible (donc sans fuite) ; train et test sont traités ensemble
    pour partager les catégories et les fréquences. Les variables dépendantes de la cible sont
    dans ResidualTargetEncoder, apprises dans chaque pli."""
    n = len(train)
    df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    out = pd.DataFrame(index=df.index)

    # -- Phase 1 : "Unknown" -> NaN + indicatrices
    for c in UNKNOWN_FLAG_COLS:
        out[f"is_unknown_{c}"] = (df[c] == "Unknown").astype("int8")
    wc = df["WeightClassKg"]
    out["wc_unknown"] = (wc == "Unknown").astype("int8")
    out["wc_is_plus"] = wc.str.contains("+", regex=False).astype("int8")
    out["wc_limit"] = pd.to_numeric(wc.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    for col, prefix in [("AgeClass", "ageclass"), ("BirthYearClass", "birthclass")]:
        for k, v in _class_bounds(df[col]).items():
            out[f"{prefix}_{k}"] = v

    cats = {c: df[c].mask(df[c] == "Unknown") for c in RAW_CATS}
    for name, (a, b) in {**INTERACTIONS, **FREQ_INTERACTIONS}.items():
        cats[name] = cats[a] + "_" + cats[b]

    # -- Phase 2 : features de domaine
    age, bw_raw, year = df["Age"], df["BodyweightKg"], df["MeetYear"]
    female = (df["Sex"] == "F").to_numpy()
    raw_eq = df["Equipment"].isin(["Raw", "Wraps"]).to_numpy()
    bw = np.where(female, bw_raw.clip(40, 150), bw_raw.clip(40, 210)).astype(float)

    out["Age"], out["BodyweightKg"], out["MeetYear"] = age, bw_raw, year
    out["log_bw"] = np.log(bw_raw.clip(lower=20))
    out["bw_pow23"] = bw_raw.clip(lower=20) ** (2 / 3)
    out["is_female"] = female.astype("int8")
    out["bw_x_male"] = np.where(female, 0.0, bw_raw)
    out["bw_x_female"] = np.where(female, bw_raw, 0.0)
    out["wilks_den"] = np.where(female, _poly(bw, _WILKS["F"]), _poly(bw, _WILKS["M"]))
    out["dots_den"] = np.where(female, _poly(bw, _DOTS["F"]), _poly(bw, _DOTS["M"]))
    ipf = np.zeros(len(df))
    for (sex, kind), (a, b, c) in _IPF_GL.items():
        m = (female == (sex == "F")) & (raw_eq == (kind == "raw"))
        ipf[m] = (a - b * np.exp(-c * bw_raw.clip(lower=20)))[m]
    out["ipf_gl_den"] = ipf

    out["age_sq"] = age**2
    out["age_dist_peak"] = (age - 30).abs()
    out["is_junior"] = (age < 23).astype("int8")
    out["is_master"] = (age >= 40).astype("int8")
    out["age_is_half"] = ((age * 2) % 2 == 1).astype("int8")
    out["bw_over_wc"] = bw_raw / out["wc_limit"].where(out["wc_limit"] > 0)

    known = cats["Country"].notna() & cats["MeetCountry"].notna()
    out["is_foreign"] = (cats["Country"] != cats["MeetCountry"]).astype(float).where(known)
    out["pre2000"] = (year < 2000).astype("int8")
    out["birth_year"] = year - age  # génération de l'athlète (effet distinct de l'âge et de l'année)

    # Taille de la compétition et de ses catégories, comptée sur train + test (sans la cible)
    meet = pd.Series(pd.util.hash_pandas_object(df[MEET], index=False).to_numpy())
    div = meet.astype(str) + "_" + df["Sex"] + "_" + df["Equipment"]
    for name, k in [("meet_size", meet), ("div_size", div), ("wc_size", div + "_" + wc)]:
        out[name] = k.map(k.value_counts()).astype(float)
    out["is_covid"] = year.isin([2020, 2021]).astype("int8")
    for c in FREQ_COLS:
        out[f"freq_{c}"] = cats[c].map(cats[c].value_counts()).fillna(0) / len(df)

    for c in CAT_COLS:
        out[c] = pd.Categorical(cats[c])
    if "Fed_Equip" in CAT_COLS:  # v1
        out = out.drop(columns=V2_ONLY)
    return out.iloc[:n].reset_index(drop=True), out.iloc[n:].reset_index(drop=True)


def outlier_mask(train: pd.DataFrame) -> np.ndarray:
    """Lignes du train physiquement improbables (Phase 1, option b) : True = aberrant."""
    ratio = train[TARGET] / train["BodyweightKg"]
    return ((train["BodyweightKg"] < 30) | (ratio < 1.0) | (ratio > 11.5)).to_numpy()


# ---------------------------------------------------------------- Phase 3 : cible

class TargetTransform:
    """A : TotalKg brut ; B : log(TotalKg) ; C : points DOTS (TotalKg * 500 / dénominateur DOTS).

    Pour C, la perte est pondérée par (dénominateur/500)² : minimiser l'erreur quadratique en points
    pondérée revient à minimiser l'erreur quadratique en kg (la métrique du concours)."""

    def __init__(self, kind: str = "A"):
        assert kind in ("A", "B", "C"), kind
        self.kind = kind

    def forward(self, y: np.ndarray, F: pd.DataFrame) -> np.ndarray:
        if self.kind == "B":
            return np.log(y)
        if self.kind == "C":
            return y * 500 / F["dots_den"].to_numpy()
        return y

    def weight(self, F: pd.DataFrame) -> np.ndarray | None:
        return (F["dots_den"].to_numpy() / 500) ** 2 if self.kind == "C" else None

    def inverse(self, p: np.ndarray, F: pd.DataFrame, log_var: float = 0.0) -> np.ndarray:
        """log_var : variance des résidus en log (correction du biais de retransformation, cible B)."""
        if self.kind == "B":
            return np.exp(p + log_var / 2)
        if self.kind == "C":
            return p * F["dots_den"].to_numpy() / 500
        return p


# ---------------------------------------------------------------- target encoding

class ResidualTargetEncoder:
    """Target encoding lissé du résidu y - moyenne(Sex × Equipment), donc du « niveau » de chaque
    fédération / pays / lieu à sexe et équipement donnés.

    fit_transform : les lignes d'entraînement sont encodées par une K-fold interne (jamais avec leur
    propre cible). transform : valeurs apprises sur tout le jeu d'entraînement (validation, test)."""

    def __init__(self, cols=TE_COLS, composites=TE_COMPOSITES, smoothing: float = 20.0, n_inner: int = 5,
                 seed: int = SEED):
        self.cols, self.composites = list(cols), dict(composites)
        self.smoothing, self.n_inner, self.seed = smoothing, n_inner, seed

    @property
    def names(self) -> list[str]:
        return [f"te_{c}" for c in [*self.cols, *self.composites]]

    @staticmethod
    def _codes(F: pd.DataFrame, col: str) -> np.ndarray:
        return F[col].cat.codes.to_numpy().astype(int) + 1  # 0 = NaN

    def _stats(self, F: pd.DataFrame, y: np.ndarray) -> dict:
        se = self._codes(F, "Sex_Equip")
        k = len(F["Sex_Equip"].cat.categories) + 1
        cnt = np.bincount(se, minlength=k)
        group_mean = np.where(cnt > 0, np.bincount(se, weights=y, minlength=k) / np.maximum(cnt, 1), y.mean())
        resid = y - group_mean[se]
        stats = {}
        for c in self.cols:
            codes, k = self._codes(F, c), len(F[c].cat.categories) + 1
            stats[c] = (np.bincount(codes, weights=resid, minlength=k), np.bincount(codes, minlength=k))
        for name, cols in self.composites.items():
            stats[name] = pd.Series(resid).groupby(self._keys(F, cols)).agg(["sum", "size"])
        return stats

    @staticmethod
    def _keys(F: pd.DataFrame, cols: list[str]) -> np.ndarray:
        return pd.util.hash_pandas_object(F[cols], index=False).to_numpy()

    def _apply(self, F: pd.DataFrame, stats: dict) -> pd.DataFrame:
        out = {}
        for c in self.cols:
            s, n = stats[c]
            te = s / (n + self.smoothing)
            te[0] = 0.0  # catégorie manquante -> a priori
            out[f"te_{c}"] = te[self._codes(F, c)]
        for name, cols in self.composites.items():
            m = stats[name].reindex(self._keys(F, cols))  # groupe absent du train -> a priori 0
            out[f"te_{name}"] = (m["sum"] / (m["size"] + self.smoothing)).fillna(0.0).to_numpy()
        return pd.DataFrame(out, index=F.index)

    def fit_transform(self, F: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        y = np.asarray(y, dtype=float)
        out = pd.DataFrame(index=F.index, columns=self.names, dtype=float)
        for tr, va in KFold(self.n_inner, shuffle=True, random_state=self.seed).split(F):
            out.iloc[va] = self._apply(F.iloc[va], self._stats(F.iloc[tr], y[tr])).to_numpy()
        self.stats_ = self._stats(F, y)
        return out

    def transform(self, F: pd.DataFrame) -> pd.DataFrame:
        return self._apply(F, self.stats_)
