"""Point d'entrée : uv run main.py <commande> --help

    train     entraîne un modèle en CV 5 ou 10 plis (OOF + test + fichier de soumission)
    tune      recherche d'hyperparamètres Optuna (lgbm / xgb / cat)
    ensemble  moyenne pondérée + stacking des 37 modèles de recipes.py (déjà entraînés)
    export-nn exporte les matrices des réseaux pour la recherche DRAGON (external/dragon/search.py)
    register  journalise une expérience entraînée ailleurs (DRAGON, TabICL) et écrit sa soumission
    dedup     post-traitement des doublons exacts du train (-> expérience <nom>_dedup)
    analyze   importance des features, résidus par segment, dépendances partielles
    score     saisit le score public Kaggle d'une expérience dans le journal
    report    tableau récapitulatif méthode -> RMSE CV -> RMSE publique
    reproduce relance les recettes des 37 modèles de l'ensemble (recipes.py)
"""

import argparse

import pandas as pd


def _param(s: str) -> tuple[str, object]:
    """« clé=valeur » -> (clé, valeur JSON si possible : true -> True, 0.5 -> 0.5 ; sinon chaîne)."""
    import json

    k, v = s.split("=", 1)
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def cmd_train(a):
    import numpy as np

    from powerlifting.config import PRED_DIR
    from powerlifting.cv import load_params, make_subfolds, prepare, run_cv
    from powerlifting.submit import write_submission

    if a.features == "v1":
        from powerlifting.features import use_v1_features
        use_v1_features()
    train, test, sample, F_tr, F_te, folds = prepare()
    split = make_subfolds(train, folds, a.folds // 5) if a.folds != 5 else None
    name = a.name or f"{a.model}_{a.target}" + ("_drop" if a.outliers == "drop" else "")
    params = {} if a.no_tuned else load_params(a.params or a.model)
    params.update(a.set or {})
    if params:
        print(f"Hyperparamètres tunés chargés : {params}")
    res = run_cv(name, a.model, train, test, F_tr, F_te, folds, target=a.target, outliers=a.outliers,
                 seeds=a.seeds, params=params, split=split, save=not a.check, only_folds=a.check_folds)
    if a.check:
        _compare_with_saved(name, res["oof"], train)
    else:
        print("Soumission :", write_submission(name, np.load(PRED_DIR / f"{name}.npy"), test, sample))


def _compare_with_saved(name, oof, train):
    """Compare une OOF recalculée à celle de outputs/oof/<name>.npy, sur les lignes recalculées."""
    import numpy as np

    from powerlifting.config import OOF_DIR, TARGET
    from powerlifting.cv import rmse

    ref = np.load(OOF_DIR / f"{name}.npy")
    rows = ~np.isnan(oof)
    y = train[TARGET].to_numpy(float)[rows]
    diff = float(np.abs(oof[rows] - ref[rows]).max())
    print(f"  [{name}] contrôle sur {rows.sum()} lignes : écart max {diff:.3g} kg ; "
          f"RMSE {rmse(y, oof[rows]):.4f} (recalculée) contre {rmse(y, ref[rows]):.4f} (sauvegardée)")
    print(f"  [{name}] {'REPRODUIT' if diff < 1e-6 else 'DIFFÉRENT'}")


def cmd_reproduce(a):
    """Relance les recettes de recipes.py, un processus par modèle (PyTorch et LightGBM / CatBoost / XGBoost ne
    cohabitent pas dans un même processus sur macOS : conflit OpenMP)."""
    import shlex
    import subprocess
    import sys

    from powerlifting.config import ROOT
    from powerlifting.recipes import RECIPES

    names = a.names or list(RECIPES)
    unknown = sorted(set(names) - set(RECIPES))
    if unknown:
        raise SystemExit(f"recettes inconnues : {unknown}")
    for n in names:
        if RECIPES[n] is None:
            print(f"# {n} : entraîné hors du pipeline, voir external/ et le README")
            continue
        cmd = [sys.executable, "main.py", "train", *shlex.split(RECIPES[n]), "--name", n]
        if a.check:
            cmd.append("--check")
        if a.check_folds:
            cmd += ["--check-folds", *map(str, a.check_folds)]
        print("$", shlex.join(cmd[1:]), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT)


def cmd_tune(a):
    from powerlifting.tune import tune

    tune(a.model, a.trials, a.folds, a.target, a.outliers)


def cmd_ensemble(a):
    from powerlifting.cv import prepare
    from powerlifting.ensemble import run_ensemble
    from powerlifting.recipes import ENSEMBLE

    names = a.models or ENSEMBLE
    print(f"Ensemble de {len(names)} modèles")
    train, test, sample, F_tr, F_te, folds = prepare()
    run_ensemble(names, train, test, sample, F_tr, F_te, folds, contrib=a.contrib)


def cmd_export_nn(a):
    from powerlifting.cv import prepare
    from powerlifting.external import export_nn_data

    train, _, _, F_tr, F_te, folds = prepare()
    export_nn_data(train, F_tr, F_te, folds, a.target)


def cmd_register(a):
    import numpy as np

    from powerlifting.config import PRED_DIR
    from powerlifting.cv import prepare
    from powerlifting.external import register_external
    from powerlifting.submit import write_submission

    train, test, sample, F_tr, _, folds = prepare()
    register_external(a.name, a.model, train, folds, target=a.target, seeds=a.seeds, minutes=a.minutes,
                      n_features=F_tr.shape[1])
    print("Soumission :", write_submission(a.name, np.load(PRED_DIR / f"{a.name}.npy"), test, sample))


def cmd_dedup(a):
    import numpy as np

    from powerlifting.config import PRED_DIR
    from powerlifting.cv import prepare
    from powerlifting.postprocess import dedup_experiment
    from powerlifting.submit import write_submission

    train, test, sample, _, _, folds = prepare()
    for name in a.names:
        dedup_experiment(name, train, test, folds)
        print("Soumission :", write_submission(f"{name}_dedup", np.load(PRED_DIR / f"{name}_dedup.npy"), test, sample))


def cmd_analyze(a):
    from powerlifting.analysis import run_analysis

    run_analysis(a.oof)


def cmd_score(a):
    from powerlifting.experiment_log import record_public_score

    record_public_score(a.name, a.score)


def cmd_report(a):
    from powerlifting.experiment_log import read_log

    log = read_log()
    cols = ["name", "model", "target", "outliers", "cv_rmse", "cv_std", "public_score", "minutes"]
    print(log[cols].sort_values("cv_rmse").to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="CV 5 ou 10 plis d'un modèle")
    t.add_argument("--model", required=True, choices=["const", "hgb", "ridge", "rf", "et", "lgbm", "cat", "xgb", "mlp", "tabm",
                                                        "realmlp"])
    t.add_argument("--target", default="A", choices=["A", "B", "C"], help="A: TotalKg, B: log, C: points DOTS")
    t.add_argument("--outliers", default="keep", choices=["keep", "drop"])
    t.add_argument("--seeds", type=int, default=1, help="nombre de seeds moyennées (GBM / MLP)")
    t.add_argument("--name", help="nom de l'expérience (défaut : modèle_cible)")
    t.add_argument("--no-tuned", action="store_true", help="ignorer outputs/params/<modèle>.json")
    t.add_argument("--folds", type=int, default=5, choices=[5, 10],
                   help="plis d'entraînement (10 : chaque pli de référence coupé en deux ; scores toujours sur 5)")
    t.add_argument("--set", nargs="+", type=_param, metavar="CLÉ=VALEUR",
                   help="hyperparamètres ajoutés aux paramètres tunés (ex. : extra_trees=true)")
    t.add_argument("--params", help="fichier outputs/params/<PARAMS>.json à la place de <modèle>.json (ex. : lgbm_v1)")
    t.add_argument("--features", default="v2", choices=["v1", "v2"],
                   help="v1 : features du 28/09, pour réentraîner les anciens modèles (journal, section 24)")
    t.add_argument("--check", action="store_true",
                   help="ne rien écrire : comparer l'OOF obtenue à outputs/oof/<nom>.npy")
    t.add_argument("--check-folds", type=int, nargs="+", metavar="K",
                   help="avec --check : seulement ces plis d'entraînement (sous-plis 0-9 si --folds 10)")
    t.set_defaults(fn=cmd_train)

    rp = sub.add_parser("reproduce", help="relancer les recettes des 37 modèles de l'ensemble (recipes.py)")
    rp.add_argument("names", nargs="*", help="recettes à relancer (défaut : toutes)")
    rp.add_argument("--dry-run", action="store_true", help="afficher les commandes sans les lancer")
    rp.add_argument("--check", action="store_true", help="comparer aux OOF sauvegardées sans rien écrire")
    rp.add_argument("--check-folds", type=int, nargs="+", metavar="K", help="avec --check : plis à recalculer")
    rp.set_defaults(fn=cmd_reproduce)

    u = sub.add_parser("tune", help="tuning Optuna")
    u.add_argument("--model", required=True, choices=["lgbm", "xgb", "cat", "rf", "et", "mlp"])
    u.add_argument("--trials", type=int, default=30)
    u.add_argument("--folds", type=int, default=3, help="nombre de plis utilisés pendant le tuning")
    u.add_argument("--target", default="A", choices=["A", "B", "C"])
    u.add_argument("--outliers", default="keep", choices=["keep", "drop"])
    u.set_defaults(fn=cmd_tune)

    e = sub.add_parser("ensemble", help="moyenne pondérée + stacking")
    e.add_argument("--models", nargs="+", help="noms d'expériences (défaut : les 37 de recipes.ENSEMBLE)")
    e.add_argument("--contrib", action="store_true", help="contribution marginale de chaque modèle au stacking")
    e.set_defaults(fn=cmd_ensemble)

    x = sub.add_parser("export-nn", help="exporter les matrices des réseaux (DRAGON)")
    x.add_argument("--target", default="C", choices=["A", "C"])
    x.set_defaults(fn=cmd_export_nn)

    g = sub.add_parser("register", help="journaliser une expérience entraînée ailleurs")
    g.add_argument("name", help="outputs/oof/<name>.npy et outputs/preds/<name>.npy doivent exister (en kg)")
    g.add_argument("--model", default="dragon")
    g.add_argument("--target", default="C", choices=["A", "C"])
    g.add_argument("--seeds", type=int, default=1)
    g.add_argument("--minutes", type=float, default=float("nan"))
    g.set_defaults(fn=cmd_register)

    d = sub.add_parser("dedup", help="post-traitement des doublons exacts")
    d.add_argument("names", nargs="+", help="expériences à post-traiter (ex. : ensemble_stack lgbm_C_tuned)")
    d.set_defaults(fn=cmd_dedup)

    an = sub.add_parser("analyze", help="importance, résidus, dépendances partielles")
    an.add_argument("--oof", default="lgbm_A", help="expérience dont l'OOF sert à l'analyse des résidus")
    an.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("score", help="enregistrer un score public")
    s.add_argument("name")
    s.add_argument("score", type=float)
    s.set_defaults(fn=cmd_score)

    r = sub.add_parser("report", help="tableau récapitulatif")
    r.set_defaults(fn=cmd_report)
    return ap


def main():
    a = build_parser().parse_args()
    pd.set_option("display.width", 200)
    a.fn(a)


if __name__ == "__main__":
    main()
