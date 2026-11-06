from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OOF_DIR = OUT_DIR / "oof"
PRED_DIR = OUT_DIR / "preds"
SUB_DIR = OUT_DIR / "submissions"
PARAM_DIR = OUT_DIR / "params"
FIG_DIR = OUT_DIR / "figures"
NN_DATA_DIR = OUT_DIR / "nn_data"  # matrices exportées pour la recherche DRAGON (non versionnées)
FOLDS_FILE = DATA_DIR / "folds.csv"
EXPERIMENTS_FILE = OUT_DIR / "experiments.csv"

COMPETITION = "performances-en-powerlifting"
TARGET = "TotalKg"
ID = "id"
SEED = 42
N_FOLDS = 5

for _d in (DATA_DIR, OOF_DIR, PRED_DIR, SUB_DIR, PARAM_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
