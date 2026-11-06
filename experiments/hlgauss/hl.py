"""Perte HL-Gauss (Imani & White 2018 ; Farebrother et al. 2024, « Stop Regressing ») pour le MLP du projet.

Même réseau que `models.mlp_fn` (embeddings + PLR + 256-128-64, BatchNorm, SiLU, dropout) mais la dernière
couche sort K logits sur un histogramme régulier de la cible C standardisée. Entraînement : cross-entropy
(pondérée par w = (dénominateur/500)², comme la MSE du projet) contre la masse d'une gaussienne N(y, σ²) sur
chaque intervalle ; prédiction = espérance de l'histogramme. Option `mse` : perte mixte CE + λ·w·(E − y)².

Usage (depuis experiments/) :
    uv run python hlgauss/hl.py fold0            # variantes + référence sur le pli 0
    uv run python hlgauss/hl.py full <nom_cfg>   # 5 plis pour une config (et la référence si nom_cfg = ref)
"""

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)
_set_threads = torch.set_num_threads
torch.set_num_threads = lambda n: None  # neutralise le set_num_threads(8) de mlp_fn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import OOF_DIR, cv_loop, evaluate, load, rmse, save  # noqa: E402

import models  # noqa: E402
from config import SEED  # noqa: E402
from models import _with_te, nn_arrays  # noqa: E402
from tabnn import TabularInput, TabularMLP, weighted_rmse  # noqa: E402

MLP_JSON = json.loads((HERE.parent.parent / "outputs" / "params" / "mlp.json").read_text())


class HLGaussMLP(nn.Module):
    """TabularMLP dont la tête scalaire est remplacée par K logits ; `centers` = centres des intervalles."""

    def __init__(self, inp, hidden, dropout, edges: np.ndarray):
        super().__init__()
        base = TabularMLP(inp, hidden, dropout)
        base.mlp[-1] = nn.Linear(hidden[-1], len(edges) - 1)
        self.base = base
        e = torch.tensor(edges, dtype=torch.float32)
        self.register_buffer("edges", e)
        self.register_buffer("centers", (e[1:] + e[:-1]) / 2)

    def logits(self, x):
        return self.base.mlp(self.base.inp(x))

    def forward(self, x):  # espérance (cible standardisée)
        return torch.softmax(self.logits(x), -1) @ self.centers


def hl_target(y: torch.Tensor, edges: torch.Tensor, sigma: float) -> torch.Tensor:
    """Masse de N(y, σ²) sur chaque intervalle, renormalisée sur le support (Farebrother et al., annexe)."""
    cdf = torch.special.erf((edges[None, :] - y[:, None]) / (math.sqrt(2) * sigma))
    z = cdf[:, -1] - cdf[:, 0]
    return (cdf[:, 1:] - cdf[:, :-1]) / z[:, None]


def predict(net, X, bs=8192):
    net.eval()
    with torch.no_grad():
        out = torch.cat([net(X[i: i + bs]) for i in range(0, len(X), bs)])
    return out.numpy() * net.y_sd + net.y_mu


def fit_hl(net, Xtr, ytr, wtr, Xva, yva, wva, *, sigma, mse=0.0, lr=1e-3, weight_decay=1e-4, batch_size=1024,
           max_epochs=80, patience=8, history=None):
    """Copie de `tabnn.fit` (AdamW, ReduceLROnPlateau, early stopping sur la RMSE kg de l'espérance) avec la
    perte CE HL-Gauss pondérée (+ λ·MSE pondérée sur l'espérance si mse > 0)."""
    yt = torch.tensor((ytr - net.y_mu) / net.y_sd, dtype=torch.float32)
    w = torch.tensor(wtr / np.mean(wtr), dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, best_state, wait = np.inf, None, 0
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), batch_size):
            idx = perm[i: i + batch_size]
            if len(idx) < 2:
                continue
            lg = net.logits(Xtr[idx])
            logp = torch.log_softmax(lg, -1)
            ce = -(hl_target(yt[idx], net.edges, sigma) * logp).sum(-1)
            loss = (w[idx] * ce).mean()
            if mse:
                e = logp.exp() @ net.centers
                loss = loss + mse * (w[idx] * (e - yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = weighted_rmse(predict(net, Xva), yva, wva)
        if history is not None:
            history.append({"epoch": epoch, "val_rmse": round(score, 3), "lr": opt.param_groups[0]["lr"]})
        sched.step(score)
        if score < best:
            best, best_state, wait = score, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.n_epochs = epoch + 1
    return best


def hl_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """Signature des modèles du projet. p : K, sigma_ratio (σ = ratio × largeur) ou sigma (absolu, en écarts-
    types de la cible), margin, mse, seeds + paramètres du MLP (plr, dropout, lr, te)."""
    p = dict(p)
    K, ratio, sig_abs = p.pop("K", 100), p.pop("sigma_ratio", 0.75), p.pop("sigma", None)
    margin, seeds, off = p.pop("margin", 0.5), p.pop("seeds", 1), p.pop("seed_off", 0)
    hidden, drop, plr = p.pop("hidden", (256, 128, 64)), p.pop("dropout", 0.15), p.pop("plr", None)
    if p.pop("te", False):
        Xtr, Xva, Xte = _with_te(Xtr, ytr, Xva, Xte)
    Atr, Ava, Ate, cards, cont_idx = nn_arrays(Xtr, Xva, Xte, 5)
    Atr, Ava, Ate = torch.from_numpy(Atr), torch.from_numpy(Ava), torch.from_numpy(Ate)
    n_num = Atr.shape[1] - len(cards)
    mu, sd = float(np.mean(ytr)), float(np.std(ytr))
    z = (ytr - mu) / sd
    edges = np.linspace(z.min() - margin, z.max() + margin, K + 1)
    sigma = sig_abs if sig_abs is not None else ratio * (edges[1] - edges[0])
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        torch.manual_seed(SEED + off + s)
        net = HLGaussMLP(TabularInput(cards, n_num, cont_idx, plr), hidden, drop, edges)
        net.y_mu, net.y_sd = mu, sd
        hist = []
        t0 = time.time()
        best = fit_hl(net, Atr, ytr, wtr, Ava, yva, wva, sigma=sigma, history=hist, **p)
        print(f"    seed {s}: K={K} σ={sigma:.3f} best {best:.3f} epochs {net.n_epochs} ({time.time() - t0:.0f}s)",
              flush=True)
        pva, pte = pva + predict(net, Ava) / seeds, pte + predict(net, Ate) / seeds
    return pva, pte


def ref_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    t0 = time.time()
    p = dict(p)
    off = p.pop("seed_off", 0)
    models.SEED = SEED + off  # mlp_fn graine torch.manual_seed(SEED + s)
    try:
        out = models.mlp_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p)
    finally:
        models.SEED = SEED
    print(f"    ref ({time.time() - t0:.0f}s)", flush=True)
    return out


BASE = {**MLP_JSON, "te": True}
CONFIGS = {
    "ref": None,
    "K100": {"K": 100, "sigma_ratio": 0.75},
    "K50": {"K": 50, "sigma_ratio": 0.75},
    "K200": {"K": 200, "sigma_ratio": 0.75},
    "K100_s03": {"K": 100, "sigma": 0.3},
    "K100_mse": {"K": 100, "sigma_ratio": 0.75, "mse": 1.0},
    "K200_mse": {"K": 200, "sigma_ratio": 0.75, "mse": 1.0},
}


def run(name, only_folds):
    """name = config ou config@k (graine SEED + k)."""
    base, _, off = name.partition("@")
    cfg, off = CONFIGS[base], int(off or 0)
    t0 = time.time()
    if cfg is None:
        oof, te = cv_loop(ref_fn, {**BASE, "seed_off": off}, only_folds=only_folds)
    else:
        oof, te = cv_loop(hl_fn, {**BASE, **cfg, "seed_off": off}, only_folds=only_folds)
    return oof, te, time.time() - t0


def fold0_diag(pred0: dict[str, np.ndarray]):
    """Sur les lignes du pli 0 : RMSE, corrélation des résidus, et gain d'un stacking Ridge local (33 OOF +
    nouveau, validation imbriquée à 5 plis internes sur le pli 0) — indicateur grossier de diversité."""
    from ensemble import _meta_features
    from sklearn.linear_model import Ridge

    from common import ensemble_members

    _, _, F_tr, _, folds, y = load()
    va = folds == 0
    yv, Fv = y[va], F_tr[va].reset_index(drop=True)
    O = np.column_stack([np.load(OOF_DIR / f"{n}.npy")[va] for n in ensemble_members()])
    inner = np.random.default_rng(0).integers(0, 5, len(yv))

    def stack(Z):
        h = np.zeros(len(yv))
        for k in range(5):
            m = inner == k
            h[m] = Ridge(alpha=10.0).fit(Z[~m], yv[~m]).predict(Z[m])
        return rmse(yv, h)

    base = stack(_meta_features(O, Fv))
    ref_mlp = np.load(OOF_DIR / "mlp_C_plr_te_k10.npy")[va]
    ref_lgb = np.load(OOF_DIR / "lgbm_C_k10.npy")[va]
    rows = []
    for n, p in pred0.items():
        r = yv - p
        rows.append({
            "nom": n, "rmse_pli0": round(rmse(yv, p), 3),
            "corr_mlp_te_k10": round(float(np.corrcoef(r, yv - ref_mlp)[0, 1]), 4),
            "corr_lgbm_C_k10": round(float(np.corrcoef(r, yv - ref_lgb)[0, 1]), 4),
            "corr_ref_pli0": round(float(np.corrcoef(r, yv - pred0["ref"])[0, 1]), 4) if "ref" in pred0 else None,
            "moy_ref_50_50": round(rmse(yv, (p + pred0["ref"]) / 2), 3) if "ref" in pred0 else None,
            "stack_delta_pli0": round(stack(_meta_features(np.column_stack([O, p]), Fv)) - base, 4),
        })
    print(f"stack local pli 0 (33 modèles) : {base:.4f}")
    for r in rows:
        print(r)
    return base, rows


if __name__ == "__main__":
    mode = sys.argv[1]
    out = HERE / "preds"
    out.mkdir(exist_ok=True)
    if mode == "fold0":
        names = sys.argv[2:] or list(CONFIGS)
        _, _, _, _, folds, _ = load()
        for n in names:
            f = out / f"{n}_fold0.npy"
            if f.exists():
                print(f"{n} : déjà fait", flush=True)
                continue
            print(f"== {n}", flush=True)
            oof, _, dt = run(n, [0])
            np.save(f, oof[folds == 0])
            print(f"   {n} : {dt / 60:.1f} min", flush=True)
    elif mode == "diag0":
        pred0 = {f.stem.replace("_fold0", ""): np.load(f) for f in sorted(out.glob("*_fold0.npy"))}
        fold0_diag(pred0)
    elif mode == "full":
        n = sys.argv[2]
        oof, te, dt = run(n, None)
        tag = "hlgauss" if not n.startswith("ref") else "hlgauss_ref_mlp_te_s1"
        np.save(out / f"{tag}_oof.npy", oof)
        np.save(out / f"{tag}_test.npy", te)
        if not n.startswith("ref"):
            save("hlgauss", oof, te)
        print(f"{n} : 5 plis en {dt / 60:.1f} min", flush=True)
        print(json.dumps(evaluate(oof), ensure_ascii=False), flush=True)
