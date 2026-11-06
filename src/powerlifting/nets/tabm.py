"""TabM (Gorishniy, Kotelnikov, Babenko, ICLR 2025), journal étape 26. Appelé par `models.neural.tabm_fn`, mêmes entrées
que `mlp_fn`. Module séparé de `nets/mlp.py`, qui doit rester sans dépendance pour DRAGON.

Entrées : `models.common.with_te` (target encoding appris dans le pli, optionnel) puis `models.neural.nn_arrays`
(codes des catégories | numériques écrêtées et standardisées | indicatrices de manquants).

Réseau : [plongement des continues (PLE version B, ou périodique) | numériques restantes brutes |
catégories en one-hot (défaut TabM) ou en embeddings] -> EnsembleView(k) -> backbone TabM (BatchEnsemble,
`make_tabm_backbone`, init « normal » par blocs de feature) -> LinearEnsemble(k sorties).
Perte : MSE pondérée (w = poids de la cible C, = MSE en kg) moyennée sur les k têtes ; prédiction = moyenne des
têtes ; early stopping sur la RMSE (kg) de la prédiction moyenne sur la validation du pli.
"""

import copy
import gc
import resource
import time

import numpy as np
import rtdl_num_embeddings as rne
import tabm
import torch
from torch import nn


class OneHot(nn.Module):
    def __init__(self, cards):
        super().__init__()
        self.cards = list(cards)

    def forward(self, codes):
        return torch.cat([nn.functional.one_hot(codes[:, i], c).float() for i, c in enumerate(self.cards)], 1)


class TabMNet(nn.Module):
    def __init__(self, cards, n_num, cont_idx, Atr_num, *, k=32, n_blocks=2, d_block=512, dropout=0.1,
                 emb="ple", n_bins=48, d_emb=16, sigma=0.1, n_freq=16, cat="onehot", arch="tabm"):
        super().__init__()
        self.n_cat = len(cards)
        other = [i for i in range(n_num) if i not in set(cont_idx)]
        self.register_buffer("cont", torch.tensor(cont_idx, dtype=torch.long))
        self.register_buffer("other", torch.tensor(other, dtype=torch.long))
        nc = len(cont_idx)
        if emb == "ple":
            bins = rne.compute_bins(Atr_num[:, cont_idx], n_bins=n_bins)
            self.num = rne.PiecewiseLinearEmbeddings(bins, d_emb, activation=False, version="B")
        elif emb == "periodic":  # PLR de l'article sur les embeddings (version rtdl, non « lite »)
            self.num = rne.PeriodicEmbeddings(nc, d_emb, n_frequencies=n_freq, frequency_init_scale=sigma, lite=False)
        elif emb == "none":
            self.num = None
        else:
            raise ValueError(emb)
        d_feat = [d_emb] * nc if self.num is not None else [1] * nc
        d_feat += [1] * len(other)
        if cat == "onehot":
            self.cat = OneHot(cards)
            d_feat += list(cards)
        else:  # embeddings comme le MLP du projet
            self.cat = nn.ModuleList([nn.Embedding(c, min(24, (c + 1) // 2 + 1)) for c in cards])
            d_feat += [e.embedding_dim for e in self.cat]
        self.ens = tabm.EnsembleView(k=k)
        self.backbone = tabm.make_tabm_backbone(
            d_in=sum(d_feat), n_blocks=n_blocks, d_block=d_block, dropout=dropout, activation="ReLU", k=k,
            arch_type=arch, start_scaling_init=None if arch == "tabm-packed" else ("normal" if self.num is not None else "random-signs"),
            start_scaling_init_chunks=None if arch == "tabm-packed" else d_feat)
        self.out = tabm.LinearEnsemble(d_block, 1, k=k)

    def forward(self, x):
        codes, num = x[:, : self.n_cat].long(), x[:, self.n_cat:]
        xc = num[:, self.cont]
        parts = [(xc if self.num is None else self.num(xc)).flatten(1), num[:, self.other]]
        if isinstance(self.cat, OneHot):
            parts.append(self.cat(codes))
        else:
            parts += [e(codes[:, i]) for i, e in enumerate(self.cat)]
        h = self.backbone(self.ens(torch.cat(parts, 1)))
        return self.out(h).squeeze(-1)  # (B, k)


def _predict(net, X, dev, bs=2048):
    net.eval()
    with torch.no_grad():
        out = torch.cat([net(X[i: i + bs].to(dev)).mean(1).cpu() for i in range(0, len(X), bs)])
    return out.numpy() * net.y_sd + net.y_mu


def _mem(dev):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # octets sous macOS
    gpu = f" mps {torch.mps.driver_allocated_memory() / 2**30:.2f} Go" if dev.type == "mps" else ""
    return f"[rss max {rss:.2f} Go{gpu}]"


def _wrmse(p, y, w):
    return float(np.sqrt(np.mean(w * (p - y) ** 2)))


def fit_tabm(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """Paramètres (p) : te, min_count, device, lr, weight_decay, batch_size, max_epochs, patience, seed,
    + ceux de TabMNet (k, n_blocks, d_block, dropout, emb, n_bins, d_emb, sigma, n_freq, cat, arch)."""
    from powerlifting.models.common import with_te
    from powerlifting.models.neural import nn_arrays

    p = dict(p)
    torch.set_num_threads(p.pop("threads", 2))
    dev = torch.device(p.pop("device", "cpu"))
    lr, wd = p.pop("lr", 2e-3), p.pop("weight_decay", 3e-4)
    bs, max_ep, patience = p.pop("batch_size", 512), p.pop("max_epochs", 100), p.pop("patience", 16)
    seed, min_count, log = p.pop("seed", 42), p.pop("min_count", 5), p.pop("log", print)
    hist = p.pop("history", None)
    if p.pop("te", True):
        Xtr, Xva, Xte = with_te(Xtr, ytr, Xva, Xte)
    Atr, Ava, Ate, cards, cont_idx = nn_arrays(Xtr, Xva, Xte, min_count)
    Atr, Ava, Ate = torch.from_numpy(Atr), torch.from_numpy(Ava), torch.from_numpy(Ate)
    n_cat = len(cards)
    torch.manual_seed(seed)
    net = TabMNet(cards, Atr.shape[1] - n_cat, cont_idx, Atr[:, n_cat:], **p).to(dev)
    net.y_mu, net.y_sd = float(np.mean(ytr)), float(np.std(ytr))
    yt = torch.tensor((ytr - net.y_mu) / net.y_sd, dtype=torch.float32)
    w = torch.tensor(wtr / np.mean(wtr), dtype=torch.float32)
    Xd, yd, wd_ = Atr.to(dev), yt.to(dev), w.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    best, best_state, best_ep, wait, t0 = np.inf, None, -1, 0, time.time()
    for ep in range(max_ep):
        net.train()
        perm = torch.randperm(len(yt), device=dev)
        for i in range(0, len(perm), bs):
            idx = perm[i: i + bs]
            if len(idx) < 2:
                continue
            out = net(Xd[idx])  # (B, k) : chaque tête voit le même batch (share_training_batches=True)
            loss = (wd_[idx, None] * (out - yd[idx, None]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ep == 0 and (i // bs) % 40 == 0:
                if dev.type == "mps":
                    torch.mps.synchronize()
                log(f"      step {i // bs} {(time.time() - t0):.1f} s  {_mem(dev)}")
        score = _wrmse(_predict(net, Ava, dev), yva, wva)
        if hist is not None:
            hist.append((ep, score))
        if score < best:
            best, best_state, best_ep, wait = score, copy.deepcopy(net.state_dict()), ep, 0
        else:
            wait += 1
        log(f"    ep {ep:3d} val {score:.3f} best {best:.3f}@{best_ep}  {(time.time() - t0) / 60:.1f} min  {_mem(dev)}")
        if wait >= patience:
            break
    net.load_state_dict(best_state)
    out = _predict(net, Ava, dev), _predict(net, Ate, dev)
    del net, opt, best_state, Xd, yd, wd_, Atr, Ava, Ate  # libère la mémoire entre les plis
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    return out
