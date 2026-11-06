"""Recherche d'architecture du MLP avec DRAGON (algorithme Mutant-UCB).

Tourne dans l'environnement séparé external/dragon/ (DRAGON impose numpy < 2) ; les données arrivent déjà prétraitées :

    uv run main.py export-nn --target C                                # matrices des 5 plis (projet principal)
    uv run --project external/dragon python external/dragon/search.py search --iters 400  # recherche sur le pli 0
    uv run --project external/dragon python external/dragon/search.py final --seeds 5     # ré-entraîne les meilleures, 5 plis
    uv run main.py register dragon_C --seeds 5                         # journal + soumission (projet principal)

Espace de recherche : l'entrée (embeddings + PLR, `powerlifting.nets.mlp.TabularInput`) et la sortie (Linear -> 1) sont fixes ;
DRAGON cherche le graphe acyclique entre les deux (nœuds : couche dense, normalisation, dropout ou identité ;
entrées combinées par somme, produit ou concaténation ; activation de chaque nœud) ainsi que le learning rate.

Mutant-UCB : chaque évaluation est un entraînement partiel de EPOCHS_PER_EVAL époques qui reprend les poids de la
configuration (stockés dans ses nœuds) ; une mutation hérite des poids de son parent. Les architectures et leurs
scores sont journalisés dans external/dragon/runs/archs.jsonl, et `final` ré-entraîne les meilleures depuis zéro.
"""

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]  # external/dragon/search.py -> racine du dépôt
sys.path.insert(0, str(ROOT / "src"))  # seul powerlifting.nets.mlp est importé (numpy + torch, sans pandas)
from dragon.search_algorithm.mutant_ucb import Mutant_UCB  # noqa: E402
from dragon.search_operators.base_neighborhoods import ArrayInterval, ConstantInterval, FloatInterval  # noqa: E402
from dragon.search_space.base_variables import ArrayVar, Constant, FloatVar  # noqa: E402
from dragon.search_space.bricks import MLP  # noqa: E402
from dragon.search_space.bricks_variables import dag_var, dropout, identity_var, mlp_var, norm_1d, operations_var  # noqa: E402
from dragon.search_space.dag_encoding import AdjMatrix, Brick, Node  # noqa: E402

from powerlifting.nets.mlp import TabularInput, fit, predict, weighted_rmse  # noqa: E402

DATA_DIR = ROOT / "outputs" / "nn_data"
RUN_DIR = ROOT / "external" / "dragon" / "runs"
ARCH_DIR = RUN_DIR / "archs"
ARCH_LOG = RUN_DIR / "archs.jsonl"
PLR = {"sigma": 0.1}  # réglage retenu pour le MLP (outputs/params/mlp.json)
EPOCHS_PER_EVAL = 4
SEED = 42


def load_fold(k: int) -> dict:
    d = dict(np.load(DATA_DIR / f"fold{k}.npz"))
    for key in ("Xtr", "Xva", "Xte"):
        d[key] = torch.from_numpy(d[key])
    d["cards"], d["cont_idx"] = d["cards"].tolist(), d["cont_idx"].tolist()
    return d


class TabInput(Brick):
    """Bloc d'entrée fixe sous forme de brique DRAGON : ses poids sont stockés et hérités avec la configuration."""

    def __init__(self, input_shape, cards, n_num, cont_idx, plr):
        super().__init__(input_shape)
        self.inp = TabularInput(cards, n_num, cont_idx, plr)

    def forward(self, X):
        return self.inp(X)

    def modify_operation(self, input_shape):
        pass


class Net(nn.Module):
    """Méta-architecture : entrée fixe -> graphe cherché par DRAGON -> Linear(1)."""

    def __init__(self, cfg: dict, n_features: int):
        super().__init__()
        self.inp, self.dag, self.out = cfg["In"], cfg["Dag"], cfg["Out"]
        self.inp.set((n_features,))
        self.dag.set(self.inp.output_shape)
        self.out.set(self.dag.output_shape)

    def forward(self, x):
        return self.out(self.dag(self.inp(x))).squeeze(-1)


def input_hp(d: dict) -> dict:
    """Dimensions du bloc d'entrée : propres à chaque pli (vocabulaire des catégories appris sur son train)."""
    return {"cards": d["cards"], "n_num": d["Xtr"].shape[1] - len(d["cards"]), "cont_idx": d["cont_idx"], "plr": PLR}


def search_space(d: dict) -> ArrayVar:
    inp = Constant("In", Node("add", TabInput, input_hp(d), nn.Identity()), neighbor=ConstantInterval())
    ops = operations_var("Operations", size=6, candidates=[
        mlp_var("Dense", max_int=512), identity_var("Identity"), dropout("Dropout"), norm_1d("Norm"),
    ])
    dag = dag_var("Dag", ops, complexity=4)
    # entrée et sortie en Constant : ArrayInterval ne les tire pas au sort pour une mutation
    out = Constant("Out", Node("add", MLP, {"out_channels": 1}, nn.Identity()), neighbor=ConstantInterval())
    log_lr = FloatVar("LogLr", lower=-3.5, upper=-2.3, neighbor=FloatInterval(0.15))
    return ArrayVar(inp, dag, out, log_lr, label="Search Space", neighbor=ArrayInterval())


def describe(dag: AdjMatrix) -> list[str]:
    return [f"{n.combiner}:{n.name.__name__}{n.hp or ''}->{type(n.activation).__name__}" for n in dag.operations]


def fresh(cfg: dict, d: dict | None = None) -> dict:
    """Même architecture, poids non initialisés (pour ré-entraîner depuis zéro), entrée dimensionnée pour le pli d."""
    clone = {k: Node(n.combiner, n.name, n.hp, n.activation) for k, n in (("In", cfg["In"]), ("Out", cfg["Out"]))}
    if d is not None:
        clone["In"].hp = input_hp(d)
    clone["Dag"] = AdjMatrix([Node(n.combiner, n.name, n.hp, n.activation) for n in cfg["Dag"].operations],
                             cfg["Dag"].matrix.copy())
    clone["LogLr"] = cfg["LogLr"]
    return clone


def run_search(iters: int, population: int, max_partial: int, explore: float):
    d = load_fold(0)
    space = search_space(d)
    labels = [v.label for v in space]
    shutil.rmtree(ARCH_DIR, ignore_errors=True)  # nouvelle recherche : on repart d'un journal vide
    ARCH_LOG.unlink(missing_ok=True)
    ARCH_DIR.mkdir(parents=True)
    torch.set_num_threads(8)
    np.random.seed(SEED)
    calls = {}

    def evaluation(args, idx):
        cfg = dict(zip(labels, args))
        calls[idx] = calls.get(idx, 0) + 1
        torch.manual_seed(SEED + 1000 * int(idx) + calls[idx])
        t0 = time.time()
        net = Net(cfg, d["Xtr"].shape[1])
        loss = fit(net, d["Xtr"], d["ytr"], d["wtr"], d["Xva"], d["yva"], d["wva"], lr=10 ** cfg["LogLr"],
                   max_epochs=EPOCHS_PER_EVAL, patience=EPOCHS_PER_EVAL)
        if calls[idx] == 1:
            with open(ARCH_DIR / f"{idx}.pkl", "wb") as f:
                pickle.dump(fresh(cfg), f)
        row = {"idx": int(idx), "call": calls[idx], "rmse": round(loss, 4), "lr": round(10 ** cfg["LogLr"], 6),
               "n_params": sum(p.numel() for p in net.parameters()), "seconds": round(time.time() - t0, 1),
               "matrix": cfg["Dag"].matrix.tolist(), "nodes": describe(cfg["Dag"])}
        with open(ARCH_LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{idx:4d}#{calls[idx]}] RMSE {loss:7.3f}  {row['seconds']:5.1f}s  lr {row['lr']:.1e}  {row['nodes'][1:]}",
              flush=True)
        return loss, net

    algo = Mutant_UCB(space, T=iters, K=population, N=max_partial, E=explore, evaluation=evaluation,
                      save_dir=str(RUN_DIR / "mutant_ucb"))
    algo.run()


def best_architectures(top: int) -> list[tuple[int, float, int]]:
    """Meilleures architectures du journal : (idx, meilleure RMSE vue en entraînement partiel, nb d'évaluations)."""
    best, n = {}, {}
    for line in ARCH_LOG.read_text().splitlines():
        r = json.loads(line)
        best[r["idx"]] = min(best.get(r["idx"], np.inf), r["rmse"])
        n[r["idx"]] = n.get(r["idx"], 0) + 1
    return [(i, best[i], n[i]) for i in sorted(best, key=best.get)[:top]]


def train_from_scratch(cfg: dict, d: dict, seed: int) -> Net:
    torch.manual_seed(seed)
    c = fresh(cfg, d)
    net = Net(c, d["Xtr"].shape[1])
    fit(net, d["Xtr"], d["ytr"], d["wtr"], d["Xva"], d["yva"], d["wva"], lr=10 ** c["LogLr"])
    return net


def run_final(top: int, seeds: int, name: str):
    torch.set_num_threads(8)
    candidates = best_architectures(top)
    d0 = load_fold(0)
    scored = []
    print(f"Ré-entraînement depuis zéro des {len(candidates)} meilleures architectures sur le pli 0 (1 seed) :")
    for idx, partial, n_eval in candidates:
        with open(ARCH_DIR / f"{idx}.pkl", "rb") as f:
            cfg = pickle.load(f)
        net = train_from_scratch(cfg, d0, SEED)
        r = weighted_rmse(predict(net, d0["Xva"]), d0["yva"], d0["wva"])
        scored.append((r, idx, cfg))
        print(f"  idx {idx:4d}  partiel {partial:7.3f} ({n_eval} éval.)  depuis zéro {r:7.3f}  "
              f"{sum(p.numel() for p in net.parameters())} paramètres  {describe(cfg['Dag'])[1:]}", flush=True)
    r0, idx, cfg = min(scored, key=lambda t: t[0])
    print(f"Retenue : idx {idx} (pli 0 : {r0:.3f}). Matrice : {cfg['Dag'].matrix.tolist()}  lr {10 ** cfg['LogLr']:.2e}")

    n_train = sum(len(np.load(DATA_DIR / f"fold{k}.npz")["va_idx"]) for k in range(5))
    oof, pred_te, t0 = np.full(n_train, np.nan), 0.0, time.time()
    for k in range(5):
        d = d0 if k == 0 else load_fold(k)
        p_va, p_te = 0.0, 0.0
        for s in range(seeds):
            net = train_from_scratch(cfg, d, SEED + s)
            p_va, p_te = p_va + predict(net, d["Xva"]) / seeds, p_te + predict(net, d["Xte"]) / seeds
        oof[d["va_idx"]] = p_va * d["scale_va"]
        pred_te += p_te * d["scale_te"] / 5
        print(f"  pli {k} : RMSE {weighted_rmse(p_va, d['yva'], d['wva']):.3f}  ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
    np.save(ROOT / "outputs" / "oof" / f"{name}.npy", oof)
    np.save(ROOT / "outputs" / "preds" / f"{name}.npy", pred_te)
    (RUN_DIR / f"{name}_architecture.txt").write_text(
        f"idx {idx}\nlr {10 ** cfg['LogLr']:.3e}\nmatrix {cfg['Dag'].matrix.tolist()}\n" + "\n".join(describe(cfg["Dag"])) + "\n")
    print(f"Écrit outputs/oof/{name}.npy et outputs/preds/{name}.npy en {(time.time() - t0) / 60:.1f} min. "
          f"Ensuite : uv run main.py register {name} --seeds {seeds} --minutes {(time.time() - t0) / 60:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="recherche Mutant-UCB sur le pli 0")
    s.add_argument("--iters", type=int, default=400, help="nombre total d'évaluations (T)")
    s.add_argument("--population", type=int, default=20, help="population initiale aléatoire (K)")
    s.add_argument("--max-partial", type=int, default=8, help="entraînements partiels max par configuration (N)")
    s.add_argument("--explore", type=float, default=1.0, help="exploration UCB (E), à l'échelle de la RMSE en kg")
    f = sub.add_parser("final", help="ré-entraîne les meilleures architectures puis la meilleure sur 5 plis")
    f.add_argument("--top", type=int, default=5)
    f.add_argument("--seeds", type=int, default=5)
    f.add_argument("--name", default="dragon_C")
    a = ap.parse_args()
    if a.cmd == "search":
        run_search(a.iters, a.population, a.max_partial, a.explore)
    else:
        run_final(a.top, a.seeds, a.name)


if __name__ == "__main__":
    main()
