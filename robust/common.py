"""Shared loading, spectral indices, and metrics."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CLASSES = ["unburned", "low", "moderate", "high"]
IGNORE = 255


def fires_table() -> pd.DataFrame:
    return pd.read_csv(DATA / "fires.csv", dtype={"pre_id": str, "post_id": str})


def load_fire(name: str) -> dict:
    d = np.load(DATA / "fires" / f"{name}.npz")
    return {"pre": d["pre"].astype(np.float32), "post": d["post"].astype(np.float32),
            "label": d["label"], "inside": d["inside"],
            "valid": d["pre_ok"] & d["post_ok"]}


# ── Spectral indices (NIR = band index 3, SWIR2 = band index 5) ──────────────

def nbr(img: np.ndarray) -> np.ndarray:
    nir, swir2 = img[3], img[5]
    return (nir - swir2) / np.where(np.abs(nir + swir2) < 1e-6, np.nan, nir + swir2)


def indices(pre: np.ndarray, post: np.ndarray) -> dict[str, np.ndarray]:
    """dNBR, RdNBR (Miller and Thode 2007), RBR (Parks et al. 2014), all x1000."""
    nbr_pre, nbr_post = nbr(pre), nbr(post)
    dnbr = (nbr_pre - nbr_post) * 1000
    rdnbr = dnbr / np.sqrt(np.maximum(np.abs(nbr_pre), 1e-3))
    rbr = dnbr / (nbr_pre + 1.001)
    return {"nbr_pre": nbr_pre, "nbr_post": nbr_post, "dnbr": dnbr, "rdnbr": rdnbr, "rbr": rbr}


# ── Metrics ──────────────────────────────────────────────────────────────────

def confusion(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    m = label != IGNORE
    return np.bincount(label[m].astype(np.int64) * 4 + pred[m].astype(np.int64), minlength=16).reshape(4, 4)


def scores(cm: np.ndarray) -> dict:
    """Per-class IoU, mIoU, accuracy, macro-F1. Classes absent from both
    prediction and label are left out of the means (not scored as 0)."""
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (tp + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    out = {f"iou_{c}": float(v) for c, v in zip(CLASSES, iou)}
    out["miou"] = float(np.nanmean(iou))
    out["macro_f1"] = float(np.nanmean(f1))
    out["acc"] = float(tp.sum() / cm.sum())
    out["n"] = int(cm.sum())
    return out
