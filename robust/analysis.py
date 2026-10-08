"""
Additional analyses for the paper, computed from saved predictions.

  pooled confusion matrices and per-class precision / recall / F1
  share of errors that confuse adjacent severity classes
  error rate as a function of distance to the nearest class boundary
  calibration gap (oracle minus learned) against threshold distance
  mean mIoU as a function of a single global high-severity threshold
  dataset statistics per fire and per fold

Writes paper/numbers_analysis.tex, paper/tables/*.tex and paper/figs/*.pdf.

Usage:
  python -m robust.analysis
"""
import json
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

from robust.common import CLASSES, DATA, IGNORE, RESULTS, ROOT, fires_table, indices, load_fire

PAPER = ROOT / "paper"
TABLES, FIGS = PAPER / "tables", PAPER / "figs"
METHODS = {"dnbr_generic": "dNBR, generic", "dnbr_learned": "dNBR, learned",
           "unet_bands_idx": "U-Net (17 ch.)", "oracle": "Oracle"}
BINS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, np.inf)]  # pixel distance to nearest other class
BIN_LABELS = ["$\\leq$30\\,m", "30--60\\,m", "60--120\\,m", "120--240\\,m", "$>$240\\,m"]

C_GENERIC, C_LEARNED, C_UNET, C_ORACLE = "#2a78d6", "#1baf7a", "#eb6834", "#8a8985"
COLORS = {"dnbr_generic": C_GENERIC, "dnbr_learned": C_LEARNED, "unet_bands_idx": C_UNET, "oracle": C_ORACLE}
INK2, GRID = "#52514e", "#e4e3df"
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "font.family": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
WIDTH = 6.5  # article class, 1 in margins

macros: dict[str, str] = {}


def macro(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = str(value)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pred(method: str, fire: str) -> np.ndarray:
    return np.load(RESULTS / "preds" / method / f"{fire}.npy")


def boundary_distance(label: np.ndarray) -> np.ndarray:
    """Distance (pixels) from each labelled pixel to the nearest labelled pixel of another class."""
    out = np.full(label.shape, np.inf, dtype=np.float32)
    for c in range(4):
        other = (label != c) & (label != IGNORE)
        if not other.any():
            continue
        d = distance_transform_edt(~other)
        m = label == c
        out[m] = d[m]
    return out


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    fires = fires_table()
    cms = {m: np.zeros((4, 4), np.int64) for m in METHODS}
    dist_err = {m: np.zeros((len(BINS), 2), np.int64) for m in METHODS}  # (errors, total) per bin
    per_fire_rows = []
    sweep_hist = []  # per fire (4, n_bins) dNBR histograms for the threshold sweep
    lo_edge, hi_edge = -500, 1500

    for f in fires.itertuples():
        d = load_fire(f.fire)
        lab = d["label"]
        m = lab != IGNORE
        dist = boundary_distance(lab)
        for meth in METHODS:
            p = pred(meth, f.fire)
            p = np.where(p == IGNORE, 0, p)
            cms[meth] += np.bincount(lab[m].astype(np.int64) * 4 + p[m], minlength=16).reshape(4, 4)
            wrong = (p != lab) & m
            for k, (a, b) in enumerate(BINS):
                sel = m & (dist > a) & (dist <= b)
                dist_err[meth][k] += [wrong[sel].sum(), sel.sum()]
        ix = indices(d["pre"], d["post"])
        x = ix["dnbr"][m]; y = lab[m]; ok = np.isfinite(x)
        h = np.zeros((4, hi_edge - lo_edge))
        np.add.at(h, (y[ok], np.clip(np.round(x[ok]), lo_edge, hi_edge - 1).astype(int) - lo_edge), 1)
        sweep_hist.append(h)
        share = np.bincount(lab[m], minlength=4) / m.sum()
        npz = np.load(DATA / "fires" / f"{f.fire}.npz")
        per_fire_rows.append({"fire": f.fire, "name": f.incid_name, "year": f.year, "fold": f.fold,
                              "acres": f.burnbndac, "pixels": int(m.sum()), **{c: s for c, s in zip(CLASSES, share)},
                              "pre": str(npz["pre_item"]), "post": str(npz["post_item"]),
                              "substituted": bool(npz["substituted"]) if "substituted" in npz.files else False,
                              "low_t": f.low_t, "mod_t": f.mod_t, "high_t": f.high_t})
        log(f"{f.fire} done")

    # ── pooled per-class precision / recall / F1 ──
    lines = []
    for meth, name in METHODS.items():
        cm = cms[meth]; tp = np.diag(cm)
        prec, rec = tp / cm.sum(0), tp / cm.sum(1)
        f1 = 2 * prec * rec / (prec + rec)
        cells = " & ".join(f"{100*p:.1f} & {100*r:.1f} & {100*f:.1f}" for p, r, f in zip(prec, rec, f1))
        lines.append(f"{name} & {cells} \\\\")
        key = "".join(ch for ch in meth.title() if ch.isalpha())
        for c, p_, r_ in zip(CLASSES, prec, rec):
            macro(f"a{key}Prec{c.title()}", f"{100*p_:.1f}"); macro(f"a{key}Rec{c.title()}", f"{100*r_:.1f}")
        off = cm.sum() - tp.sum()
        adjacent = sum(cm[i, j] for i in range(4) for j in range(4) if abs(i - j) == 1)
        macro(f"a{key}AdjShare", f"{100 * adjacent / off:.1f}")
        macro(f"a{key}OverShare", f"{100 * sum(cm[i, j] for i in range(4) for j in range(4) if j > i) / off:.1f}")
    (TABLES / "per_class.tex").write_text(r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{l""" + "ccc" * 4 + r"""}
\toprule
 & \multicolumn{3}{c}{Unburned} & \multicolumn{3}{c}{Low} & \multicolumn{3}{c}{Moderate} & \multicolumn{3}{c}{High} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}
Method & P & R & F1 & P & R & F1 & P & R & F1 & P & R & F1 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Generic thresholds lose moderate-severity recall to the high class, and calibration restores it.} Pooled precision (P), recall (R), and F1 (\%) per class over all labeled pixels of the 31 held-out fires.}
\label{tab:perclass}
\end{table}
""")

    # ── confusion matrices figure ──
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.7))
    for ax, (meth, name) in zip(axes, METHODS.items()):
        cm = cms[meth]; frac = cm / cm.sum(1, keepdims=True)
        ax.imshow(frac, cmap=BLUES, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{100*frac[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if frac[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(4), ["U", "L", "M", "H"]); ax.set_yticks(range(4), ["U", "L", "M", "H"])
        ax.set_title(name, fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("MTBS class")
    for ax in axes:
        ax.set_xlabel("Predicted")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIGS / "confusion.pdf"); plt.close(fig)

    # ── error rate vs distance to class boundary ──
    fig, ax = plt.subplots(figsize=(WIDTH * 0.55, 2.0))
    x = np.arange(len(BINS))
    for meth in ("dnbr_generic", "dnbr_learned", "unet_bands_idx"):
        e = dist_err[meth]; rate = e[:, 0] / e[:, 1]
        ax.plot(x, rate, marker="o", ms=4, lw=1.6, color=COLORS[meth], label=METHODS[meth])
    shares = dist_err["dnbr_learned"][:, 1] / dist_err["dnbr_learned"][:, 1].sum()
    ax.set_xticks(x, [l.replace("\\,", " ").replace("--", "–") for l in BIN_LABELS], rotation=0)
    ax.set_xlabel("Distance to nearest pixel of another MTBS class")
    ax.set_ylabel("Pixel error rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIGS / "boundary.pdf"); plt.close(fig)
    for k, s in enumerate(shares):
        macro(f"aBinShare{'ABCDE'[k]}", f"{100*s:.1f}")
    for meth in ("dnbr_generic", "dnbr_learned", "unet_bands_idx", "oracle"):
        e = dist_err[meth]; rate = e[:, 0] / e[:, 1]
        key = "".join(ch for ch in meth.title() if ch.isalpha())
        macro(f"a{key}ErrNear", f"{100*rate[0]:.1f}"); macro(f"a{key}ErrFar", f"{100*rate[-1]:.1f}")
        macro(f"a{key}ErrNearShare", f"{100 * e[0, 0] / e[:, 0].sum():.1f}")

    # ── calibration gap vs threshold distance ──
    metrics = pd.concat([pd.read_csv(p) for p in sorted(RESULTS.glob("metrics_*.csv"))])
    piv = metrics.pivot_table(index="fire", columns="method", values="miou")
    learned = json.loads((RESULTS / "baseline_settings.json").read_text())
    fdf = fires.set_index("fire")
    dist_thr = pd.Series({f: float(np.abs(np.array([r.low_t, r.mod_t, r.high_t]) - np.array(learned[str(int(r.fold))]["dnbr"])).sum())
                          for f, r in fdf.iterrows()})
    gap = piv["oracle"] - piv["dnbr_learned"]
    rho, p = spearmanr(dist_thr.loc[gap.index], gap)
    macro("aGapRho", f"{rho:.2f}"); macro("aGapP", f"{p:.3f}" if p >= 0.001 else "<0.001")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.42, 2.0))
    ax.scatter(dist_thr.loc[gap.index], gap, s=14, color=C_LEARNED, edgecolors="white", linewidths=0.5)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel(r"$\sum_k |\tau^{\mathrm{analyst}}_k - \tau^{\mathrm{learned}}_k|$ (dNBR)")
    ax.set_ylabel("Oracle mIoU $-$ learned mIoU")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.savefig(FIGS / "calibration_gap.pdf"); plt.close(fig)

    # ── single global high threshold sweep (low/moderate fixed at the mean learned values) ──
    arr = np.array([v["dnbr"] for v in learned.values()])
    t_low, t_mod = arr[:, 0].mean(), arr[:, 1].mean()
    edges = np.arange(lo_edge, hi_edge + 1)
    cums = [np.concatenate([np.zeros((4, 1)), np.cumsum(h, 1)], 1) for h in sweep_hist]
    highs = np.arange(380, 761, 10)
    curve = []
    for th in highs:
        idx = np.searchsorted(edges, [t_low, t_mod, th])
        bounds = [0, *idx, sweep_hist[0].shape[1]]
        vals = []
        for c in cums:
            cm = np.stack([c[:, bounds[k + 1]] - c[:, bounds[k]] for k in range(4)], 1)
            tp = np.diag(cm); den = cm.sum(0) + cm.sum(1) - tp
            with np.errstate(invalid="ignore", divide="ignore"):
                vals.append(np.nanmean(tp / den))
        curve.append(np.mean(vals))
    curve = np.array(curve)
    best = highs[curve.argmax()]
    macro("aSweepBest", int(best)); macro("aSweepBestMiou", f"{curve.max():.3f}")
    macro("aSweepAtGeneric", f"{curve[np.argmin(np.abs(highs - 440))]:.3f}")
    macro("aSweepLow", f"{t_low:.0f}"); macro("aSweepMod", f"{t_mod:.0f}")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.42, 2.0))
    ax.plot(highs, curve, color=C_LEARNED, lw=1.8)
    ax.axvline(440, color=C_GENERIC, lw=1, ls=(0, (3, 2)))
    ax.text(446, curve.min(), "generic 440", fontsize=7, color=C_GENERIC, va="bottom")
    ax.axvline(best, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("High-severity threshold (dNBR)")
    ax.set_ylabel("Mean mIoU over fires")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.savefig(FIGS / "threshold_sweep.pdf"); plt.close(fig)

    # ── dataset table (appendix) ──
    df = pd.DataFrame(per_fire_rows).sort_values(["fold", "acres"], ascending=[True, False])
    body = []
    for r in df.itertuples():
        pretty = {"MCFARLAND": "McFarland", "MCCASH": "McCash", "KNP COMPLEX": "KNP Complex",
                  "CZU AUG LIGHTNING": "CZU Lightning"}.get(r.name, r.name.title())
        scene = lambda s: s.split("_")[0][:4] + " " + s.split("_")[3][:4] + "-" + s.split("_")[3][4:6] + "-" + s.split("_")[3][6:]
        body.append(f"{pretty}{'$^\\dagger$' if r.substituted else ''} & {r.year} & {r.fold} & {r.acres/1000:.0f} & "
                    f"{r.pixels/1e6:.2f} & {100*r.unburned:.0f} / {100*r.low:.0f} / {100*r.moderate:.0f} / {100*r.high:.0f} & "
                    f"{int(r.low_t)} / {int(r.mod_t)} / {int(r.high_t)} & {scene(r.pre)} & {scene(r.post)} \\\\")
    (TABLES / "fires.tex").write_text(r"""\begin{table}[H]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccccc}
\toprule
Fire & Year & Fold & Acres (k) & Px (M) & U / L / M / H (\%) & Analyst $\tau$ & Pre-fire scene & Post-fire scene \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{The 31 fires.} Px is the number of labeled pixels inside the MTBS perimeter after masking. U, L, M, and H give the share of labeled pixels in each class. Analyst $\tau$ lists the MTBS dNBR thresholds (unburned/low, low/moderate, moderate/high). Scenes are Landsat Collection~2 Level-2 (sensor, acquisition date). $^\dagger$The archive does not contain the exact MTBS post-fire scene, and the closest Landsat 9 scene of the same path and row, acquired 8 days later, replaces it.}
\label{tab:fires}
\end{table}
""")
    folds = df.groupby("fold").agg(n=("fire", "size"), acres=("acres", "sum"), px=("pixels", "sum"))
    macro("aPxMin", f"{df.pixels.min()/1e6:.2f}"); macro("aPxMax", f"{df.pixels.max()/1e6:.2f}")
    macro("aHighShareMin", f"{100*df.high.min():.0f}"); macro("aHighShareMax", f"{100*df.high.max():.0f}")
    macro("aFoldFiresMin", int(folds.n.min())); macro("aFoldFiresMax", int(folds.n.max()))
    macro("aFoldAcresMin", f"{folds.acres.min()/1e6:.2f}"); macro("aFoldAcresMax", f"{folds.acres.max()/1e6:.2f}")

    # ── learned thresholds per fold (appendix) ──
    rows = []
    for fold, v in sorted(learned.items(), key=lambda kv: int(kv[0])):
        cells = " & ".join(" / ".join(str(t) for t in v[k]) for k in ("dnbr", "rdnbr", "rbr"))
        rows.append(f"{fold} & {cells} \\\\")
    (TABLES / "learned.tex").write_text(r"""\begin{table}[H]
\centering
\small
\begin{tabular}{cccc}
\toprule
Held-out fold & dNBR & RdNBR & RBR \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Learned thresholds are stable across folds.} Thresholds (unburned/low, low/moderate, moderate/high) fit on the four training folds for each held-out fold.}
\label{tab:learned}
\end{table}
""")

    # ── training time per step, from the U-Net logs ──
    secs = []
    for log_path in [RESULTS / "log_unet_bands_idx.txt", RESULTS / "log_unet_bands.txt"]:
        if log_path.exists():
            secs += [float(x) for x in re.findall(r"\(([0-9.]+) s/step\)", log_path.read_text())[-5:]]
    if secs:
        macro("aSecPerStep", f"{np.median(secs):.2f}")
        macro("aMinPerFold", f"{np.median(secs) * 3000 / 60:.0f}")

    # ── alignment: agreement of dNBR + analyst thresholds with MTBS under one-pixel shifts, every fire ──
    shifted, gains, shifted18, shifted21 = 0, [], 0, 0
    for f in fires.itertuples():
        d = load_fire(f.fire); lab = d["label"]; m = lab != IGNORE
        q = np.digitize(np.nan_to_num(indices(d["pre"], d["post"])["dnbr"], nan=-1e9), [f.low_t, f.mod_t, f.high_t])
        g = {(dy, dx): 100 * np.mean(np.roll(q, (dy, dx), (0, 1))[m] == lab[m]) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        best = max(g, key=g.get)
        if best != (0, 0):
            shifted += 1; gains.append(g[best] - g[(0, 0)])
            shifted18 += int(f.year == 2018); shifted21 += int(f.year == 2021)
    macro("aAlignShifted", shifted); macro("aAlignShiftedEighteen", shifted18)
    macro("aAlignNEighteen", int((fires.year == 2018).sum()))
    macro("aAlignShiftedTwentyOne", shifted21)
    macro("aAlignGainMin", f"{min(gains):.1f}" if gains else "0"); macro("aAlignGainMax", f"{max(gains):.1f}" if gains else "0")

    # ── size ratio between the largest and smallest fire (labeled pixels) ──
    macro("aPxRatio", f"{df.pixels.max() / df.pixels.min():.0f}")

    # ── flat region of the threshold sweep: within 0.01 mIoU of the peak ──
    flat = highs[curve >= curve.max() - 0.01]
    macro("aSweepFlatLo", int(flat.min())); macro("aSweepFlatHi", int(flat.max()))

    # ── dNBR distance to the nearest analyst threshold, near vs far from class transitions ──
    near_d, far_d = [], []
    for f in fires.itertuples():
        d = load_fire(f.fire); lab = d["label"]; m = lab != IGNORE
        x = indices(d["pre"], d["post"])["dnbr"]
        dist = boundary_distance(lab)
        gap_t = np.min(np.abs(x[..., None] - np.array([f.low_t, f.mod_t, f.high_t])), axis=-1)
        ok = m & np.isfinite(gap_t)
        near_d.append(gap_t[ok & (dist <= 1)]); far_d.append(gap_t[ok & (dist > 8)])
    macro("aThrGapNear", f"{np.median(np.concatenate(near_d)):.0f}")
    macro("aThrGapFar", f"{np.median(np.concatenate(far_d)):.0f}")

    lines = ["% Generated by robust/analysis.py. Do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (PAPER / "numbers_analysis.tex").write_text("\n".join(lines) + "\n")
    log(f"wrote {len(macros)} macros")


if __name__ == "__main__":
    main()
