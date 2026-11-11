"""Tests du pipeline : plis, absence de fuite, cohérence OOF/journal, soumissions, repères."""

import numpy as np
import pandas as pd
import pytest

from powerlifting import submit
from powerlifting.config import ID, N_FOLDS, OOF_DIR, TARGET
from powerlifting.cv import make_folds, rmse, run_cv
from powerlifting.experiment_log import read_log
from powerlifting.features import ResidualTargetEncoder, TargetTransform, build_features


@pytest.fixture(scope="module")
def data():
    from powerlifting.data import load_data

    train, test, sample = load_data()
    F_tr, F_te = build_features(train, test)
    return train, test, sample, F_tr, F_te


# ---------------------------------------------------------------- plis

def test_folds_partition_the_train_set(data):
    train = data[0]
    folds = make_folds(train)
    assert len(folds) == len(train)
    assert set(folds) == set(range(N_FOLDS))
    sizes = np.bincount(folds)
    assert sizes.max() - sizes.min() <= 1
    # chaque pli de validation est disjoint des autres et leur union est tout le train
    parts = [set(train[ID][folds == k]) for k in range(N_FOLDS)]
    assert sum(len(p) for p in parts) == len(set().union(*parts)) == len(train)


def test_subfolds_are_nested_in_the_reference_folds(data):
    from powerlifting.cv import make_subfolds

    train = data[0]
    folds = make_folds(train)
    sub = make_subfolds(train, folds, 2)
    assert np.array_equal(sub // 2, folds), "chaque sous-pli doit rester dans son pli de référence"
    assert set(sub) == set(range(2 * N_FOLDS))
    sizes = np.bincount(sub)
    assert sizes.max() - sizes.min() <= 2
    assert np.array_equal(sub, make_subfolds(train, folds, 2)), "découpage non déterministe"


def test_folds_are_fixed_and_stratified(data):
    train = data[0]
    folds = make_folds(train)
    assert np.array_equal(folds, make_folds(train)), "les plis doivent être identiques d'un appel à l'autre"
    means = [train[TARGET][folds == k].mean() for k in range(N_FOLDS)]
    assert max(means) - min(means) < 3.0, "plis mal stratifiés sur la cible"


# ---------------------------------------------------------------- pas de fuite

def test_features_do_not_contain_the_target(data):
    _, _, _, F_tr, F_te = data
    assert TARGET not in F_tr.columns and TARGET not in F_te.columns
    assert list(F_tr.columns) == list(F_te.columns)


def test_target_encoding_never_uses_the_row_own_target(data):
    """Modifier la cible d'une ligne ne change pas SON encodage (K-fold interne) mais change celui des
    autres lignes de la même fédération."""
    _, _, _, F_tr, _ = data
    F = F_tr.iloc[:20000].reset_index(drop=True)
    y = data[0][TARGET].to_numpy(float)[:20000]
    i = int(F.index[F["Federation"] == F["Federation"].mode()[0]][0])
    y2 = y.copy()
    y2[i] += 5000.0

    te1 = ResidualTargetEncoder().fit_transform(F, y)
    te2 = ResidualTargetEncoder().fit_transform(F, y2)
    assert np.allclose(te1.iloc[i], te2.iloc[i]), "fuite : l'encodage d'une ligne dépend de sa propre cible"
    others = (F["Federation"] == F["Federation"].iloc[i]) & (F.index != i)
    assert not np.allclose(te1.loc[others, "te_Federation"], te2.loc[others, "te_Federation"])


def test_nn_arrays_are_fitted_on_the_training_fold(data):
    """Matrices des réseaux (partagées avec DRAGON) : pas de NaN, codes < cardinalité, et une fédération absente du
    train est codée 0 (« rare / inconnue ») : le vocabulaire ne voit pas la validation."""
    from powerlifting.features import CAT_COLS
    from powerlifting.models.neural import nn_arrays

    _, _, _, F_tr, F_te = data
    Xtr, Xva = F_tr.iloc[:20000].reset_index(drop=True), F_tr.iloc[20000:40000].reset_index(drop=True)
    Atr, Ava, Ate, cards, cont_idx = nn_arrays(Xtr, Xva, F_te)
    for A in (Atr, Ava, Ate):
        assert A.dtype == np.float32 and not np.isnan(A).any()
        assert (A[:, : len(cards)] < np.array(cards)).all()
    unseen = (Xva["Federation"].notna() & ~Xva["Federation"].isin(set(Xtr["Federation"].dropna()))).to_numpy()
    assert unseen.any()
    assert (Ava[unseen, CAT_COLS.index("Federation")] == 0).all()
    assert cont_idx and max(cont_idx) < A.shape[1] - len(cards)


def test_target_encoding_of_validation_ignores_validation_targets(data):
    train, _, _, F_tr, _ = data
    y = train[TARGET].to_numpy(float)
    folds = make_folds(train)
    tr, va = np.where(folds != 0)[0], np.where(folds == 0)[0]
    te = ResidualTargetEncoder()
    te.fit_transform(F_tr.iloc[tr].reset_index(drop=True), y[tr])
    a = te.transform(F_tr.iloc[va].reset_index(drop=True))
    # les statistiques sont figées après fit : transformer deux fois donne le même résultat
    assert np.allclose(a, te.transform(F_tr.iloc[va].reset_index(drop=True)))


@pytest.mark.parametrize("kind", ["A", "B", "C"])
def test_target_transform_round_trip(data, kind):
    train, _, _, F_tr, _ = data
    y = train[TARGET].to_numpy(float)
    tt = TargetTransform(kind)
    assert np.allclose(tt.inverse(tt.forward(y, F_tr), F_tr), y)


# ---------------------------------------------------------------- OOF / journal

def test_oof_files_match_the_experiment_log(data):
    """La RMSE OOF recalculée depuis les fichiers doit correspondre au journal."""
    train = data[0]
    y, folds = train[TARGET].to_numpy(float), make_folds(train)
    log = read_log()
    checked = 0
    for _, row in log.iterrows():
        f = OOF_DIR / f"{row['name']}.npy"
        if not f.exists():
            continue
        oof = np.load(f)
        assert not np.isnan(oof).any(), f"{row['name']} : OOF incomplète"
        per_fold = [rmse(y[folds == k], oof[folds == k]) for k in range(N_FOLDS)]
        assert np.mean(per_fold) == pytest.approx(row["cv_rmse"], abs=2e-2), row["name"]
        checked += 1
    if checked == 0:
        pytest.skip("aucune expérience dans le journal")


def test_reference_scores(data):
    """Repères : constante ≈ 171, HGB brut ≈ 79."""
    train, test, _, F_tr, F_te = data
    folds = make_folds(train)
    const = run_cv("test_const", "const", train, test, F_tr, F_te, folds, save=False, verbose=False)
    assert const["cv_rmse"] == pytest.approx(171, abs=1.5)
    log = read_log().set_index("name")
    if "hgb_raw" in log.index:
        assert 77.5 < log.loc["hgb_raw", "cv_rmse"] < 80.5


# ---------------------------------------------------------------- soumissions

def _good_submission(sample):
    return pd.DataFrame({ID: sample[ID], TARGET: 400.0})


def test_check_submission_accepts_valid_file(data):
    sample = data[2]
    submit.check_submission(_good_submission(sample), sample)


@pytest.mark.parametrize("break_it", ["short", "nan", "range", "ids", "columns"])
def test_check_submission_rejects_invalid_files(data, break_it):
    sample = data[2]
    sub = _good_submission(sample)
    if break_it == "short":
        sub = sub.iloc[:-1]
    elif break_it == "nan":
        sub.loc[3, TARGET] = np.nan
    elif break_it == "range":
        sub.loc[3, TARGET] = 5000.0
    elif break_it == "ids":
        sub.loc[3, ID] = -1
    elif break_it == "columns":
        sub = sub.rename(columns={TARGET: "pred"})
    with pytest.raises(AssertionError):
        submit.check_submission(sub, sample)


def test_write_submission_reorders_and_clips(data, tmp_path, monkeypatch):
    _, test, sample, _, _ = data
    monkeypatch.setattr(submit, "SUB_DIR", tmp_path)
    pred = np.full(len(test), 5000.0)
    path = submit.write_submission("t", pred, test, sample)
    out = pd.read_csv(path)
    assert len(out) == 45000 and out[ID].tolist() == sample[ID].tolist()
    assert out[TARGET].between(20, 1400).all()


# ---------------------------------------------------------------- post-traitement des doublons

def test_duplicates_are_replaced_by_the_reference_mean_only(data):
    """Une ligne identique à des lignes de référence reçoit leur moyenne ; les autres gardent leur prédiction, et la
    cible de la ligne elle-même n'intervient jamais (seules les lignes de référence sont utilisées)."""
    from powerlifting.postprocess import replace_duplicates, row_keys

    train = data[0]
    ref = train.iloc[[0, 0, 1]]  # deux lignes de référence identiques à la requête 0
    queries = pd.concat([train.iloc[[0]], train.iloc[[10]]], ignore_index=True)
    queries.loc[0, TARGET] = 99999.0  # cible de la requête : ne doit pas compter
    ref_y = np.array([400.0, 500.0, 600.0])
    out, hit = replace_duplicates(np.array([1.0, 2.0]), row_keys(queries), row_keys(ref), ref_y)
    assert hit.tolist() == [True, False]
    assert out.tolist() == [450.0, 2.0]


# ---------------------------------------------------------------- étape 26 : TabPrep, ensemble soumis

def test_tabprep_statistics_come_from_the_training_fold_only(data):
    """Variables relatives TabPrep : 345 colonnes alignées sur les index, et modifier des lignes de validation ne
    change ni les colonnes du train ni celles des autres lignes de validation (statistiques figées sur le train)."""
    from powerlifting.tabprep import tabprep_features

    _, _, _, F_tr, F_te = data
    Xtr, Xva = F_tr.iloc[:20000].reset_index(drop=True), F_tr.iloc[20000:30000].reset_index(drop=True)
    Xte = F_te.iloc[:5000].reset_index(drop=True)
    a_tr, a_va, a_te = tabprep_features(Xtr, Xva, Xte)
    assert a_tr.shape == (len(Xtr), 345) and a_va.shape[1] == a_te.shape[1] == 345
    assert a_va.index.equals(Xva.index) and a_te.index.equals(Xte.index)
    Xva2 = Xva.copy()
    Xva2.loc[:4999, ["BodyweightKg", "Age"]] = Xva2.loc[:4999, ["BodyweightKg", "Age"]].to_numpy()[::-1]
    b_tr, b_va, _ = tabprep_features(Xtr, Xva2, Xte)
    pd.testing.assert_frame_equal(a_tr, b_tr)
    pd.testing.assert_frame_equal(a_va.iloc[5000:], b_va.iloc[5000:])


def test_submitted_ensemble_members_have_predictions():
    from powerlifting.config import PRED_DIR
    from powerlifting.recipes import ENSEMBLE

    assert len(ENSEMBLE) == len(set(ENSEMBLE)) == 37
    for n in ENSEMBLE:
        assert (OOF_DIR / f"{n}.npy").exists() and (PRED_DIR / f"{n}.npy").exists(), n


def test_recipes_point_to_known_models():
    """Chaque recette est une commande `main.py train` valide sur un modèle du registre."""
    import shlex

    from main import build_parser
    from powerlifting.models import MODELS
    from powerlifting.recipes import RECIPES

    parser = build_parser()
    for name, args in RECIPES.items():
        if args is None:
            continue
        a = parser.parse_args(["train", *shlex.split(args), "--name", name])
        assert a.model in MODELS, name


def test_sibling_pairs_never_use_the_target_of_the_corrected_row(data):
    """Correction « candidats même athlète » : les références d'une ligne sont des lignes du train d'un autre pli,
    et aucune variable d'une paire (a, b) ne dépend de la cible des lignes du pli de a."""
    from powerlifting import siblings as sb

    train, test = data[0].iloc[:12000].reset_index(drop=True), data[1].iloc[:3000].reset_index(drop=True)
    folds = np.arange(len(train)) % N_FOLDS
    rng = np.random.default_rng(0)
    oof, pred = train[TARGET].to_numpy(float) + rng.normal(0, 70, len(train)), np.full(len(test), 500.0)
    al = sb.stack_frame(train, test, folds, oof, pred)
    pairs = sb.candidate_pairs(al)
    P = sb.pair_table(al, pairs)
    assert len(P) > 1000
    assert (P["fold_b"] >= 0).all(), "une référence doit être une ligne du train"
    assert (P["fold_b"] != P["fold_a"]).all(), "une référence ne doit pas être dans le pli de la ligne corrigée"
    al2 = al.copy()
    al2.loc[al2["fold"] == 0, "res"] += 1000.0  # on change la cible de toutes les lignes du pli 0
    P2 = sb.pair_table(al2, pairs)
    a0 = (P["fold_a"] == 0).to_numpy()
    feats = [c for c in P.columns if c != "res_a"]
    pd.testing.assert_frame_equal(P.loc[a0, feats], P2.loc[a0, feats])
