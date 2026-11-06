"""Briques PyTorch du MLP tabulaire (M7), partagées par `models.py` et la recherche d'architecture DRAGON
(`dragon/search.py`). Ce module ne dépend que de numpy et torch pour tourner aussi dans l'environnement
séparé de DRAGON (qui impose numpy < 2).

Entrée des réseaux : une seule matrice float `[codes des catégories | numériques standardisées | indicatrices
de manquants]`, construite par `models.nn_arrays`.
"""

import copy

import numpy as np
import torch
from torch import nn


class PeriodicEmbedding(nn.Module):
    """Plongement « PLR » des variables continues (Gorishniy et al., 2022, *On Embeddings for Numerical
    Features in Tabular Deep Learning*) : x -> ReLU(Linear([cos 2πcx, sin 2πcx])), une couche linéaire par
    variable et des fréquences c ~ N(0, sigma) apprises. Donne au MLP des fonctions de base périodiques,
    là où une entrée brute ne permet que des découpages quasi linéaires."""

    def __init__(self, n: int, k: int = 16, d: int = 16, sigma: float = 0.5):
        super().__init__()
        self.n, self.k = n, k
        # fréquences en 1-D : DRAGON ré-initialise (xavier) les paramètres de dimension > 1, pas celui-ci
        self.freq = nn.Parameter(torch.randn(n * k) * sigma)
        self.weight = nn.Parameter(torch.empty(n, 2 * k, d).uniform_(-(2 * k) ** -0.5, (2 * k) ** -0.5))
        self.bias = nn.Parameter(torch.zeros(n, d))
        self.out_dim = n * d

    def forward(self, x):
        v = 2 * torch.pi * x[..., None] * self.freq.view(self.n, self.k)
        v = torch.cat([torch.cos(v), torch.sin(v)], dim=-1)
        return torch.relu(torch.einsum("bnk,nkd->bnd", v, self.weight) + self.bias).flatten(1)


class TabularInput(nn.Module):
    """Embeddings des catégories (dimension min(24, (k+1)/2 + 1)) + numériques, brutes ou passées en PLR pour
    les colonnes `cont_idx` (variables continues ; les indicatrices restent brutes)."""

    def __init__(self, cards: list[int], n_num: int, cont_idx: list[int], plr: dict | None = None):
        super().__init__()
        self.n_cat = len(cards)
        self.embs = nn.ModuleList([nn.Embedding(k, min(24, (k + 1) // 2 + 1)) for k in cards])
        emb_dim = sum(e.embedding_dim for e in self.embs)
        if plr:
            self.register_buffer("cont", torch.tensor(cont_idx, dtype=torch.long))
            self.register_buffer("other", torch.tensor([i for i in range(n_num) if i not in set(cont_idx)], dtype=torch.long))
            self.plr = PeriodicEmbedding(len(cont_idx), **plr)
            self.out_dim = emb_dim + self.plr.out_dim + len(self.other)
        else:
            self.plr = None
            self.out_dim = emb_dim + n_num

    def forward(self, x):
        codes, num = x[:, : self.n_cat].long(), x[:, self.n_cat :]
        parts = [e(codes[:, i]) for i, e in enumerate(self.embs)]
        if self.plr is None:
            parts.append(num)
        else:
            parts += [self.plr(num[:, self.cont]), num[:, self.other]]
        return torch.cat(parts, dim=1)


class TabularMLP(nn.Module):
    def __init__(self, inp: TabularInput, hidden=(256, 128, 64), dropout: float = 0.15):
        super().__init__()
        self.inp = inp
        d, layers = inp.out_dim, []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.mlp = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, x):
        return self.mlp(self.inp(x)).squeeze(-1)


def predict(net: nn.Module, X: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
    """Prédiction dans l'espace de la cible (le réseau sort la cible standardisée par `fit`)."""
    net.eval()
    with torch.no_grad():
        out = torch.cat([net(X[i : i + batch_size]).reshape(-1) for i in range(0, len(X), batch_size)])
    return out.numpy() * net.y_sd + net.y_mu


def weighted_rmse(p: np.ndarray, y: np.ndarray, w: np.ndarray | None) -> float:
    """sqrt(moyenne(w·e²)) : avec w = (dénominateur DOTS / 500)², c'est exactement la RMSE en kg (cible C)."""
    e2 = (p - y) ** 2
    return float(np.sqrt(np.mean(e2 if w is None else w * e2)))


def fit(
    net: nn.Module, Xtr, ytr, wtr, Xva, yva, wva, *, lr: float = 2e-3, weight_decay: float = 1e-4,
    batch_size: int = 1024, max_epochs: int = 80, patience: int = 8, history: list | None = None,
) -> float:
    """Entraîne `net` (AdamW, learning rate divisé par 2 sur plateau, early stopping sur la validation) et
    le laisse dans son meilleur état. Retourne la RMSE de validation (en kg pour les cibles A et C).

    history : si une liste est donnée, on y ajoute à chaque époque la RMSE sur le train (sous-échantillon de
    20 000 lignes, réseau en mode évaluation) et sur la validation, pour diagnostiquer le sur-apprentissage."""
    net.y_mu, net.y_sd = float(np.mean(ytr)), float(np.std(ytr))
    yt = torch.tensor((ytr - net.y_mu) / net.y_sd, dtype=torch.float32)
    w = torch.ones(len(ytr)) if wtr is None else torch.tensor(wtr / np.mean(wtr), dtype=torch.float32)
    probe = np.random.default_rng(0).choice(len(ytr), min(20000, len(ytr)), replace=False)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, best_state, wait = np.inf, None, 0
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2:
                continue
            loss = (w[idx] * (net(Xtr[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = weighted_rmse(predict(net, Xva), yva, wva)
        if history is not None:
            p_tr = predict(net, Xtr[probe])
            history.append({"epoch": epoch, "train_rmse": weighted_rmse(p_tr, ytr[probe], None if wtr is None else wtr[probe]),
                            "val_rmse": score, "lr": opt.param_groups[0]["lr"]})
        sched.step(score)
        if score < best:
            best, best_state, wait = score, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    return best
