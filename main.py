"""Point d'entrée : uv run main.py <commande> --help

    train     entraîne un modèle en CV 5 ou 10 plis (OOF + test + fichier de soumission)
    tune      recherche d'hyperparamètres Optuna (lgbm / xgb / cat)
    ensemble  moyenne pondérée + stacking des modèles de ensemble_models.txt (déjà entraînés)
    export-nn exporte les matrices des réseaux pour la recherche DRAGON (dragon/search.py)
    register  journalise une expérience entraînée ailleurs (DRAGON) et écrit sa soumission
    dedup     post-traitement des doublons exacts du train (-> expérience <nom>_dedup)
    analyze   importance des features, résidus par segment, dépendances partielles
    score     saisit le score public Kaggle d'une expérience dans le journal
    report    tableau récapitulatif méthode -> RMSE CV -> RMSE publique
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

    from config import PRED_DIR
    from cv import load_params, make_subfolds, prepare, run_cv
    from submit import write_submission

    if a.features == "v1":
        from features import use_v1_features
        use_v1_features()
    train, test, sample, F_tr, F_te, folds = prepare()
    split = make_subfolds(train, folds, a.folds // 5) if a.folds != 5 else None
    name = a.name or f"{a.model}_{a.target}" + ("_drop" if a.outliers == "drop" else "")
    params = {} if a.no_tuned else load_params(a.params or a.model)
    params.update(a.set or {})
    if params:
        print(f"Hyperparamètres tunés chargés : {params}")
    res = run_cv(name, a.model, train, test, F_tr, F_te, folds, target=a.target, outliers=a.outliers,
                 seeds=a.seeds, params=params, split=split)
    print("Soumission :", write_submission(name, np.load(PRED_DIR / f"{name}.npy"), test, sample))


def cmd_tune(a):
    from tune import tune

    tune(a.model, a.trials, a.folds, a.target, a.outliers)


def cmd_ensemble(a):
    from cv import prepare, read_log
    from ensemble import run_ensemble

    from config import ENSEMBLE_FILE

    if a.models:
        names = a.models
    elif a.all or not ENSEMBLE_FILE.exists():
        names = [n for n in read_log()["name"]
                 if n != "const" and not n.startswith("ensemble") and not n.endswith("_dedup")]
    else:
        names = [l.strip() for l in ENSEMBLE_FILE.read_text().splitlines() if l.strip()]
    print(f"Ensemble de {len(names)} modèles")
    train, test, sample, F_tr, F_te, folds = prepare()
    run_ensemble(names, train, test, sample, F_tr, F_te, folds, contrib=a.contrib)


def cmd_export_nn(a):
    from cv import export_nn_data, prepare

    train, _, _, F_tr, F_te, folds = prepare()
    export_nn_data(train, F_tr, F_te, folds, a.target)


def cmd_register(a):
    import numpy as np

    from config import PRED_DIR
    from cv import prepare, register_external
    from submit import write_submission

    train, test, sample, F_tr, _, folds = prepare()
    register_external(a.name, a.model, train, folds, target=a.target, seeds=a.seeds, minutes=a.minutes,
                      n_features=F_tr.shape[1])
    print("Soumission :", write_submission(a.name, np.load(PRED_DIR / f"{a.name}.npy"), test, sample))


def cmd_dedup(a):
    import numpy as np

    from config import PRED_DIR
    from cv import prepare
    from postprocess import dedup_experiment
    from submit import write_submission

    train, test, sample, _, _, folds = prepare()
    for name in a.names:
        dedup_experiment(name, train, test, folds)
        print("Soumission :", write_submission(f"{name}_dedup", np.load(PRED_DIR / f"{name}_dedup.npy"), test, sample))


def cmd_analyze(a):
    from analysis import run_analysis

    run_analysis(a.oof)


def cmd_score(a):
    from cv import record_public_score

    record_public_score(a.name, a.score)


def cmd_report(a):
    from cv import read_log

    log = read_log()
    cols = ["name", "model", "target", "outliers", "cv_rmse", "cv_std", "public_score", "minutes"]
    print(log[cols].sort_values("cv_rmse").to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
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
    t.set_defaults(fn=cmd_train)

    u = sub.add_parser("tune", help="tuning Optuna")
    u.add_argument("--model", required=True, choices=["lgbm", "xgb", "cat", "rf", "et", "mlp"])
    u.add_argument("--trials", type=int, default=30)
    u.add_argument("--folds", type=int, default=3, help="nombre de plis utilisés pendant le tuning")
    u.add_argument("--target", default="A", choices=["A", "B", "C"])
    u.add_argument("--outliers", default="keep", choices=["keep", "drop"])
    u.set_defaults(fn=cmd_tune)

    e = sub.add_parser("ensemble", help="moyenne pondérée + stacking")
    e.add_argument("--models", nargs="+", help="noms d'expériences (défaut : la liste de ensemble_models.txt)")
    e.add_argument("--all", action="store_true", help="tous les modèles du journal au lieu de ensemble_models.txt")
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

    a = ap.parse_args()
    pd.set_option("display.width", 200)
    a.fn(a)


if __name__ == "__main__":
    main()
