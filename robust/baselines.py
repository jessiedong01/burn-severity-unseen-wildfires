"""
Non-deep baselines, evaluated with the same fire-grouped 5-fold split as the U-Net.

  otsu            Gaussian smoothing (sigma 1.5) + 4-class multi-Otsu on dNBR, per fire
  dnbr_generic    dNBR with the generic thresholds 100 / 270 / 440
  dnbr_learned    dNBR thresholds fit on the training fires
  rdnbr_learned   RdNBR thresholds fit on the training fires
  rbr_learned     RBR thresholds fit on the training fires
  gbm             per-pixel gradient boosting on 12 bands + 5 indices (no spatial context)
  oracle          dNBR with the MTBS analyst's own thresholds for that fire (uses test-fire
                  information; a reference ceiling, not a competitor)

Outputs:
  results/preds/<method>/<fire>.npy     predicted class map (255 where inputs invalid)
  results/metrics_baselines.csv         per-fire scores and confusion matrices
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_multiotsu
from sklearn.ensemble import HistGradientBoostingClassifier

from robust.common import IGNORE, RESULTS, confusion, fires_table, indices, load_fire, scores

GENERIC = {"dnbr": (100, 270, 440)}  # Key and Benson (2006) class edges
SEARCH_START = {"dnbr": (100, 270, 440), "rdnbr": (100, 300, 600), "rbr": (100, 270, 440)}
GRID = {"dnbr": (-500, 1500), "rdnbr": (-1000, 3000), "rbr": (-400, 1200)}  # histogram range, 1-unit bins
GBM_SAMPLES_PER_FIRE = 60_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def classify(x: np.ndarray, thresholds) -> np.ndarray:
    pred = np.digitize(np.nan_to_num(x, nan=-1e9), thresholds).astype(np.uint8)
    pred[~np.isfinite(x)] = IGNORE
    return pred


def fit_thresholds(hists: list[np.ndarray], lo: int, start) -> tuple:
    """Coordinate search for 3 thresholds maximising the mean (over fires) mIoU.

    hists: per fire, a (4 classes, n_bins) count histogram of the index.
    """
    edges = np.arange(lo, lo + hists[0].shape[1] + 1)
    cums = [np.concatenate([np.zeros((4, 1)), np.cumsum(h, axis=1)], axis=1) for h in hists]

    def objective(t):
        idx = np.searchsorted(edges, t)  # bin index where each threshold falls
        bounds = [0, *idx, hists[0].shape[1]]
        vals = []
        for c in cums:
            cm = np.stack([c[:, bounds[k + 1]] - c[:, bounds[k]] for k in range(4)], axis=1)  # true x pred
            tp = np.diag(cm); denom = cm.sum(0) + cm.sum(1) - tp
            with np.errstate(invalid="ignore", divide="ignore"):
                vals.append(np.nanmean(tp / denom))
        return np.nanmean(vals)

    t = list(start)
    best = objective(t)
    for step in (64, 16, 4, 1):
        improved = True
        while improved:
            improved = False
            for i in range(3):
                for d in (-step, step):
                    cand = t.copy(); cand[i] += d
                    if not (cand[0] < cand[1] < cand[2]):
                        continue
                    v = objective(cand)
                    if v > best + 1e-9:
                        best, t, improved = v, cand, True
    return tuple(t), best


def otsu_pred(dnbr: np.ndarray, label: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(dnbr), dnbr, np.nanmedian(dnbr))
    smooth = gaussian_filter(filled, sigma=1.5)
    region = label != IGNORE
    th = threshold_multiotsu(smooth[region], classes=4)
    pred = np.digitize(smooth, th).astype(np.uint8)
    pred[~np.isfinite(dnbr)] = IGNORE
    return pred


def features(f: dict, ix: dict) -> np.ndarray:
    feats = [f["pre"][i] for i in range(6)] + [f["post"][i] for i in range(6)]
    feats += [ix[k] for k in ("nbr_pre", "nbr_post", "dnbr", "rdnbr", "rbr")]
    return np.stack(feats, axis=-1)  # (H, W, 17)


def main() -> None:
    fires = fires_table()
    rows, settings = [], {}
    cache = {}

    def get(name):
        if name not in cache:
            f = load_fire(name)
            cache[name] = (f, indices(f["pre"], f["post"]))
        return cache[name]

    # histograms per fire per index, for threshold fitting
    hist = {k: {} for k in GRID}
    for name in fires.fire:
        f, ix = get(name)
        m = f["label"] != IGNORE
        for k, (lo, hi) in GRID.items():
            x = np.clip(np.round(ix[k][m]), lo, hi - 1).astype(int) - lo
            ok = np.isfinite(ix[k][m])
            h = np.zeros((4, hi - lo))
            np.add.at(h, (f["label"][m][ok], x[ok]), 1)
            hist[k][name] = h
    log("histograms done")

    for fold in sorted(fires.fold.unique()):
        train = fires[fires.fold != fold].fire.tolist()
        test = fires[fires.fold == fold].fire.tolist()
        learned = {}
        for k in GRID:
            start = SEARCH_START[k]
            t, v = fit_thresholds([hist[k][n] for n in train], GRID[k][0], start)
            learned[k] = t
            log(f"fold {fold}: learned {k} thresholds {t} (train mean mIoU {v:.3f})")

        # gradient boosting on a per-fire sample of training pixels
        rng = np.random.default_rng(fold)
        X, y = [], []
        for n in train:
            f, ix = get(n)
            m = np.flatnonzero((f["label"] != IGNORE).ravel())
            take = rng.choice(m, size=min(GBM_SAMPLES_PER_FIRE, len(m)), replace=False)
            X.append(features(f, ix).reshape(-1, 17)[take]); y.append(f["label"].ravel()[take])
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                             class_weight="balanced", random_state=0)
        gbm.fit(np.concatenate(X), np.concatenate(y))
        log(f"fold {fold}: gbm trained on {sum(len(a) for a in y):,} pixels")
        settings[int(fold)] = {k: list(map(int, v)) for k, v in learned.items()}

        for n in test:
            f, ix = get(n)
            r = fires.set_index("fire").loc[n]
            preds = {
                "otsu": otsu_pred(ix["dnbr"], f["label"]),
                "dnbr_generic": classify(ix["dnbr"], GENERIC["dnbr"]),
                "dnbr_learned": classify(ix["dnbr"], learned["dnbr"]),
                "rdnbr_learned": classify(ix["rdnbr"], learned["rdnbr"]),
                "rbr_learned": classify(ix["rbr"], learned["rbr"]),
                "oracle": classify(ix["dnbr"], (r.low_t, r.mod_t, r.high_t)),
            }
            F = features(f, ix)
            ok = np.isfinite(F).all(-1)
            g = np.full(ok.shape, IGNORE, dtype=np.uint8)
            g[ok] = gbm.predict(F[ok]).astype(np.uint8)
            preds["gbm"] = g
            for method, p in preds.items():
                out = RESULTS / "preds" / method
                out.mkdir(parents=True, exist_ok=True)
                np.save(out / f"{n}.npy", p)
                cm = confusion(np.where(p == IGNORE, 0, p), f["label"])
                rows.append({"method": method, "fire": n, "fold": int(fold), **scores(cm),
                             "cm": json.dumps(cm.tolist())})
            log(f"fold {fold}: {n} done")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "metrics_baselines.csv", index=False)
    (RESULTS / "baseline_settings.json").write_text(json.dumps(settings, indent=2))
    print(df.groupby("method")[["miou", "macro_f1", "acc"]].mean().sort_values("miou").round(3).to_string())


if __name__ == "__main__":
    main()
