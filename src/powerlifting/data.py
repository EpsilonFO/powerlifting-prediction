import os
from pathlib import Path

# Les fichiers Kaggle sont rangés dans data/ (et non dans ~/.cache) ; à définir avant tout appel à kagglehub.
from powerlifting.config import COMPETITION, DATA_DIR

os.environ.setdefault("KAGGLEHUB_CACHE", str(DATA_DIR))

import kagglehub  # noqa: E402
import pandas as pd  # noqa: E402


def ensure_login() -> None:
    """Demande le token Kaggle une seule fois : il est ensuite sauvegardé dans ~/.kaggle/."""
    try:
        kagglehub.whoami(verbose=False)
    except Exception:
        print("Aucun identifiant Kaggle trouvé : création d'un token sur https://www.kaggle.com/settings/api")
        kagglehub.login()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Renvoie train, test, sample_submission. Les fichiers déjà téléchargés sont lus sans réseau ; sinon la
    compétition est téléchargée avec kagglehub."""
    path = DATA_DIR / "competitions" / COMPETITION
    if not all((path / f).exists() for f in ("train.csv", "test.csv", "sample_submission.csv")):
        ensure_login()
        path = Path(kagglehub.competition_download(COMPETITION))
    train = pd.read_csv(path / "train.csv")
    test = pd.read_csv(path / "test.csv")
    sample = pd.read_csv(path / "sample_submission.csv")
    return train, test, sample

