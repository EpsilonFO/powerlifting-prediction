"""Assemble les 10 sorties du kernel Kaggle (tabicl10_C_sub{j}.npz) en prédictions OOF et test du projet :
outputs/oof/tabicl_C_k10.npy et outputs/preds/tabicl_C_k10.npy (en kg). Ensuite :

    uv run python tabicl/assemble.py tabicl/out
    uv run main.py register tabicl_C_k10 --model tabicl --target C
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OOF_DIR, PRED_DIR  # noqa: E402

NAME = "tabicl_C_k10"


def main(src: Path):
    parts = [np.load(src / f"tabicl10_C_sub{j}.npz") for j in range(10)]
    n_tr = sum(len(p["va"]) for p in parts)
    oof, test = np.full(n_tr, np.nan), np.zeros(len(parts[0]["pred_te"]))
    for p in parts:
        assert np.isnan(oof[p["va"]]).all(), "sous-plis qui se chevauchent"
        oof[p["va"]] = p["pred_va"]
        test += p["pred_te"] / len(parts)
    assert not np.isnan(oof).any(), "OOF incomplète"
    np.save(OOF_DIR / f"{NAME}.npy", oof)
    np.save(PRED_DIR / f"{NAME}.npy", test)
    print(f"{NAME} : OOF {oof.shape}, test {test.shape} -> outputs/oof, outputs/preds")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "out")
