"""
Tables, numbers, and figures for the report, all generated from results/.

  paper/numbers.tex            \\newcommand macros quoted in the text
  paper/tables/main.tex        every method, mean over fires and pooled
  paper/tables/per_fire.tex    per-fire mIoU (appendix)
  paper/figs/*.pdf             figures

Usage:
  python -m robust.report
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from robust.common import CLASSES, DATA, IGNORE, RESULTS, ROOT, fires_table, load_fire, scores

REPORT = ROOT / "paper"
TABLES, FIGS = REPORT / "tables", REPORT / "figs"

LABELS = {
    "otsu": "Gaussian + multi-Otsu",
    "dnbr_generic": "dNBR, generic thresholds",
    "dnbr_learned": "dNBR, learned thresholds",
    "rdnbr_learned": "RdNBR, learned thresholds",
    "rbr_learned": "RBR, learned thresholds",
    "gbm": "Per-pixel gradient boosting",
    "unet_bands": "U-Net, 12 bands",
    "unet_bands_idx": "U-Net, 12 bands + indices",
    "oracle": "Oracle (analyst thresholds)",
}
ORDER = list(LABELS)

# Categorical slots 1-3 (validated all-pairs, light mode) + neutral gray reference.
C_GENERIC, C_LEARNED, C_UNET, C_ORACLE = "#2a78d6", "#1baf7a", "#eb6834", "#8a8985"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Ordinal severity ramp: neutral for unburned, one orange-red hue light -> dark.
SEV_COLORS = ["#e8e6e1", "#f6c9a8", "#e8794a", "#9c2f12"]
SEV_CMAP = ListedColormap(SEV_COLORS + ["#ffffff"])
SEV_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 255.5], SEV_CMAP.N)

plt.rcParams.update({
    "font.family": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL, FULL = 3.6, 6.5  # article class, 1 in margins: 6.5 in text width

macros: dict[str, str] = {}


def signed(x: float, digits: int = 3) -> str:
    """Signed number in math mode: a true minus sign that cannot break across lines."""
    return "\\ensuremath{" + f"{x:+.{digits}f}" + "}"


FIX_NAMES = {"Mcfarland": "McFarland", "Mccash": "McCash", "Knp Complex": "KNP Complex",
             "Czu Aug Lightning": "CZU Lightning Complex", "W-5 Cold Springs": "W-5 Cold Springs"}


def pretty(name: str) -> str:
    t = name.title()
    return FIX_NAMES.get(t, t)


def macro(name, value):
    assert name.isalpha(), name
    macros[name] = str(value)


def load_metrics() -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in sorted(RESULTS.glob("metrics_*.csv"))]
    return pd.concat(parts, ignore_index=True)


def best_unet(df: pd.DataFrame) -> str:
    unets = [m for m in ("unet_bands_idx", "unet_bands") if m in set(df.method)]
    return max(unets, key=lambda m: df[df.method == m].miou.mean()) if unets else None


def pooled(df_m: pd.DataFrame) -> dict:
    cm = sum(np.array(json.loads(c)) for c in df_m.cm)
    return scores(cm)


def fire_bootstrap(diff: np.ndarray, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def tables(df: pd.DataFrame, n_fires: int) -> None:
    rows = []
    for m in [m for m in ORDER if m in set(df.method)]:
        d = df[df.method == m]
        if d.fire.nunique() < n_fires:
            continue
        p = pooled(d)
        rows.append((m, d.miou.mean(), d.miou.std(), d.macro_f1.mean(), d.acc.mean(), p["miou"],
                     *[d[f"iou_{c}"].mean() for c in CLASSES]))
    best = {k: max(r[k] for r in rows if r[0] != "oracle") for k in (1, 3, 4, 5, 6, 7, 8, 9)}
    lines = []
    for r in rows:
        if r[0] == "oracle":
            lines.append(r"\midrule")
        cells = []
        for k in (1, 3, 4, 5, 6, 7, 8, 9):
            v = f"{r[k]:.3f}" if k != 1 else f"{r[1]:.3f}\\,{{\\scriptsize$\\pm${r[2]:.3f}}}"
            cells.append(rf"\textbf{{{v}}}" if r[0] != "oracle" and np.isclose(r[k], best[k]) else v)
        lines.append(f"{LABELS[r[0]]} & " + " & ".join(cells) + r" \\")
        key = r[0].replace("_", "").replace("bandsidx", "Idx").replace("bands", "Bands")
        key = "".join(ch for ch in key if ch.isalpha())
        macro(f"m{key}Miou", f"{r[1]:.3f}"); macro(f"m{key}MiouSd", f"{r[2]:.3f}")
        macro(f"m{key}Fone", f"{r[3]:.3f}"); macro(f"m{key}Acc", f"{r[4]:.3f}")
        macro(f"m{key}Pooled", f"{r[5]:.3f}")
        for c, k in zip(CLASSES, (6, 7, 8, 9)):
            macro(f"m{key}Iou{c.title()}", f"{r[k]:.3f}")
    (TABLES / "main.tex").write_text(r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{4}{c}{Mean over fires / pooled} & \multicolumn{4}{c}{IoU per class} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Method & mIoU & F1 & Acc. & Pooled & U & L & M & H \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{Results on """ + str(n_fires) + r""" held-out fires under fire-grouped five-fold cross-validation against MTBS labels, where mIoU, macro-F1 (F1), and pixel accuracy (Acc.) are averaged over fires with one standard deviation across fires ($\pm$), Pooled gives the mIoU of the summed confusion matrices, and per-class IoU is averaged over fires for the unburned (U), low (L), moderate (M), and high (H) classes. Bold marks the best value among methods trained only on other fires, and the oracle, which applies each test fire's own MTBS analyst thresholds, is listed separately as an upper reference.}
\label{tab:main}
\end{table}
""")

    piv = df.pivot_table(index="fire", columns="method", values="miou")
    fires = fires_table().set_index("fire")
    cols = [m for m in ("otsu", "dnbr_generic", "dnbr_learned", "gbm", best_unet(df), "oracle") if m in piv]
    body = []
    for f in fires.sort_values(["fold", "burnbndac"], ascending=[True, False]).index:
        if f not in piv.index:
            continue
        r = fires.loc[f]
        body.append(f"{pretty(r.incid_name)} ({r.year}) & {r.fold} & {r.burnbndac / 1000:.0f} & "
                    + " & ".join(f"{piv.loc[f, c]:.3f}" for c in cols) + r" \\")
    head = " & ".join(["Otsu", "Generic", "Learned", "GBM", "U-Net", "Oracle"][:len(cols)])
    (TABLES / "per_fire.tex").write_text(r"""\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcr""" + "c" * len(cols) + r"""}
\toprule
Fire & Fold & Acres (k) & """ + head + r""" \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\caption{Per-fire mIoU, where each fire is scored by models trained on the other four folds, Fold gives the cross-validation fold in which the fire was tested, and acres are given in thousands.}
\label{tab:perfire}
\end{table}
""")


def paired(df: pd.DataFrame) -> None:
    u = best_unet(df)
    if u is None:
        return
    piv = df.pivot_table(index="fire", columns="method", values="miou")
    if "unet_bands" in piv and "dnbr_learned" in piv:  # the other U-Net variant vs calibrated dNBR
        diff = (piv["unet_bands"] - piv["dnbr_learned"]).dropna().to_numpy()
        lo, hi = fire_bootstrap(diff)
        macro("mDiffBandsLearned", signed(diff.mean()))
        macro("mDiffBandsLearnedLo", signed(lo)); macro("mDiffBandsLearnedHi", signed(hi))
        macro("mWinsBandsLearned", int((diff > 0).sum()))
    if "unet_bands" in piv and "unet_bands_idx" in piv:
        diff = (piv["unet_bands_idx"] - piv["unet_bands"]).dropna().to_numpy()
        lo, hi = fire_bootstrap(diff)
        macro("mDiffIdxBands", signed(diff.mean()))
        macro("mDiffIdxBandsLo", signed(lo)); macro("mDiffIdxBandsHi", signed(hi))
    for ref, key in (("dnbr_learned", "Learned"), ("dnbr_generic", "Generic"), ("gbm", "Gbm")):
        if ref not in piv:
            continue
        diff = (piv[u] - piv[ref]).dropna().to_numpy()
        lo, hi = fire_bootstrap(diff)
        macro(f"mDiff{key}", signed(diff.mean()))
        macro(f"mDiff{key}Lo", signed(lo)); macro(f"mDiff{key}Hi", signed(hi))
        macro(f"mWins{key}", int((diff > 0).sum()))
    macro("mBestUnet", LABELS[u])
    macro("mBestUnetMiou", f"{piv[u].mean():.3f}")
    lt = json.loads((RESULTS / "baseline_settings.json").read_text())
    arr = np.array([v["dnbr"] for v in lt.values()])
    for i, k in enumerate(("Low", "Mod", "High")):
        macro(f"mLearned{k}Min", int(arr[:, i].min())); macro(f"mLearned{k}Max", int(arr[:, i].max()))
    # calibration: learned vs generic thresholds, and the oracle gap
    for a, b, key in (("dnbr_learned", "dnbr_generic", "LearnGen"), ("oracle", "dnbr_learned", "Oracle")):
        diff = (piv[a] - piv[b]).dropna().to_numpy()
        lo, hi = fire_bootstrap(diff)
        macro(f"mDiff{key}", signed(diff.mean()))
        macro(f"mDiff{key}Lo", signed(lo)); macro(f"mDiff{key}Hi", signed(hi))
        macro(f"mWins{key}", int((diff > 0).sum()))
    # the 12-band U-Net's first run of fold 4 diverged; it was rerun with gradient clipping
    before = RESULTS / "first_run" / "metrics_unet_bands.csv"  # first run, before the fold-4 rerun
    if before.exists():
        b = pd.read_csv(before)
        macro("mBandsFoldFourBefore", f"{b[b.fold == 4].miou.mean():.3f}")
        hist = json.loads((RESULTS / "history_unet_bands.json").read_text())
        macro("mBandsFoldFourAfter", f"{df[(df.method == 'unet_bands') & (df.fold == 4)].miou.mean():.3f}")
        macro("mBandsClip", hist["4"].get("clip"))


# ── Figures ──────────────────────────────────────────────────────────────────

def false_color(img: np.ndarray) -> np.ndarray:
    """SWIR2 / NIR / red composite, a standard burn view (burned areas look red-brown)."""
    rgb = np.stack([img[5], img[3], img[2]], -1)
    lo, hi = np.nanpercentile(rgb, 2, axis=(0, 1)), np.nanpercentile(rgb, 98, axis=(0, 1))
    return np.clip(np.nan_to_num((rgb - lo) / (hi - lo)), 0, 1)


def fig_maps(df: pd.DataFrame) -> None:
    u = best_unet(df)
    fires = fires_table()
    piv = df.pivot_table(index="fire", columns="method", values="miou")
    show = ["camp_2018"]
    if u in piv:
        ranked = piv[u].drop(index=[f for f in show if f in piv.index]).sort_values()
        show += [ranked.index[len(ranked) // 2], ranked.index[0]]  # median fire and hardest fire
    cols = [("post", "Post-fire image"), ("label", "MTBS"),
            ("dnbr_generic", "Generic dNBR"), ("dnbr_learned", "Learned dNBR")]
    if u:
        cols.append((u, "U-Net"))
    height = 1.55 * len(show) + 0.35
    fig, axes = plt.subplots(len(show), len(cols), figsize=(FULL, height), squeeze=False)
    for i, name in enumerate(show):
        f = load_fire(name)
        inside = f["inside"]
        for j, (key, title) in enumerate(cols):
            ax = axes[i, j]
            if key == "post":
                ax.imshow(false_color(f["post"]))
                ax.contour(inside, levels=[0.5], colors="white", linewidths=0.6)
            else:
                m = f["label"] if key == "label" else np.load(RESULTS / "preds" / key / f"{name}.npy")
                shown = inside & (f["label"] != IGNORE) if key == "label" else inside
                m = np.where(shown, m, IGNORE)
                ax.imshow(m, cmap=SEV_CMAP, norm=SEV_NORM, interpolation="nearest")
                if key != "label" and key in piv:
                    ax.set_xlabel(f"mIoU {piv.loc[name, key]:.2f}", fontsize=7, labelpad=1)
            if i == 0:
                ax.set_title(title, fontsize=7.5)
            if j == 0:
                r = fires.set_index("fire").loc[name]
                ax.set_ylabel(f"{pretty(r.incid_name)} ({r.year})", fontsize=7.5)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEV_COLORS] + [plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK2, lw=0.5)]
    fig.tight_layout(h_pad=0.6, w_pad=0.3, rect=(0, 0.3 / height, 1, 1))
    fig.legend(handles, [c.title() for c in CLASSES] + ["Outside perimeter / masked"], loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(FIGS / "maps.pdf", dpi=220); fig.savefig(FIGS / "maps.png", dpi=150)
    plt.close(fig)


def fig_per_fire(df: pd.DataFrame) -> None:
    u = best_unet(df)
    series = [(m, LABELS[m], c) for m, c in (("dnbr_generic", C_GENERIC), ("dnbr_learned", C_LEARNED), (u, C_UNET)) if m]
    piv = df.pivot_table(index="fire", columns="method", values="miou")
    order = piv[series[-1][0]].sort_values().index
    fires = fires_table().set_index("fire")
    fig, ax = plt.subplots(figsize=(COL, 4.2))
    y = np.arange(len(order))
    if "oracle" in piv:
        ax.scatter(piv.loc[order, "oracle"], y, marker="|", s=40, color=C_ORACLE, label=LABELS["oracle"], zorder=2)
    for m, lab, c in series:
        ax.scatter(piv.loc[order, m], y, s=16, color=c, label=lab, zorder=3, edgecolors="white", linewidths=0.6)
    ax.set_yticks(y, [f"{pretty(fires.loc[f, 'incid_name'])} '{str(fires.loc[f, 'year'])[2:]}" for f in order], fontsize=7.5)
    ax.set_xlabel("mIoU on the held-out fire")
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False, fontsize=7.5,
              handletextpad=0.2, ncol=2, columnspacing=0.8)
    fig.savefig(FIGS / "per_fire.pdf"); fig.savefig(FIGS / "per_fire.png", dpi=200)
    plt.close(fig)


def fig_thresholds() -> None:
    f = fires_table().sort_values("high_t")
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    x = np.arange(len(f))
    for col, gen, lab in (("low_t", 100, "unburned/low"), ("mod_t", 270, "low/moderate"), ("high_t", 440, "moderate/high")):
        ax.scatter(x, f[col], s=10, color=SEV_COLORS[{"low_t": 1, "mod_t": 2, "high_t": 3}[col]],
                   edgecolors=INK2, linewidths=0.4, zorder=3, label=f"Analyst, {lab}")
        ax.axhline(gen, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(len(f) + 0.3, gen, f"generic {gen}", ha="left", va="center", fontsize=6, color=INK2, clip_on=False)
    ax.set_xticks([])
    ax.set_xlabel(f"The {len(f)} fires, sorted by the analyst's high-severity threshold")
    ax.set_ylabel("dNBR threshold")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlim(-1, len(f))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6, ncol=3,
              handletextpad=0.1, columnspacing=0.8)
    fig.savefig(FIGS / "thresholds.pdf"); fig.savefig(FIGS / "thresholds.png", dpi=200)
    plt.close(fig)
    macro("mHighMin", int(f.high_t.min())); macro("mHighMax", int(f.high_t.max()))
    macro("mLowMin", int(f.low_t.min())); macro("mLowMax", int(f.low_t.max()))
    macro("mModMin", int(f.mod_t.min())); macro("mModMax", int(f.mod_t.max()))


def fig_overview() -> None:
    """The task on the Camp Fire: pre- and post-fire images, dNBR, and the MTBS map."""
    from robust.common import indices
    f = load_fire("camp_2018")
    inside = f["inside"]
    dnbr = indices(f["pre"], f["post"])["dnbr"]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.55))
    panels = [("(a) Pre-fire, Jul 2018", false_color(f["pre"])), ("(b) Post-fire, Jul 2019", false_color(f["post"])),
              ("(c) dNBR", None), ("(d) MTBS severity", None)]
    for ax, (title, img) in zip(axes, panels):
        if img is not None:
            ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    dnbr_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("dnbr", ["#f2f1ee", "#f6c9a8", "#e8794a", "#9c2f12", "#4a1408"])
    im = axes[2].imshow(np.clip(dnbr, -100, 1000), cmap=dnbr_cmap, vmin=-100, vmax=1000)
    lab = np.where(inside & (f["label"] != IGNORE), f["label"], IGNORE)
    axes[3].imshow(lab, cmap=SEV_CMAP, norm=SEV_NORM, interpolation="nearest")
    for ax in axes[:3]:
        ax.contour(inside, levels=[0.5], colors="white" if ax is not axes[2] else INK2, linewidths=0.5)
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEV_COLORS]
    axes[3].legend(handles, [c.title() for c in CLASSES], loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=6.5, handlelength=1, handletextpad=0.4)
    fig.tight_layout(w_pad=0.4)
    fig.savefig(FIGS / "overview.pdf", dpi=220)
    plt.close(fig)


def fig_summary(df: pd.DataFrame) -> None:
    """Mean mIoU over held-out fires with 95% bootstrap intervals (resampling fires)."""
    piv = df.pivot_table(index="fire", columns="method", values="miou")
    methods = [m for m in ORDER if m in piv]
    color = {"dnbr_generic": C_GENERIC, "dnbr_learned": C_LEARNED, "unet_bands": C_UNET,
             "unet_bands_idx": C_UNET, "oracle": C_ORACLE}
    fig, ax = plt.subplots(figsize=(COL, 0.24 * len(methods) + 0.45))
    for i, m in enumerate(reversed(methods)):
        v = piv[m].dropna().to_numpy()
        lo, hi = np.percentile(np.random.default_rng(0).choice(v, (10_000, len(v))).mean(1), [2.5, 97.5])
        c = color.get(m, INK2)
        ax.plot([lo, hi], [i, i], color=c, lw=1.6, solid_capstyle="round")
        ax.scatter(v.mean(), i, s=22, color=c, zorder=3, edgecolors="white", linewidths=0.6)
        ax.text(hi + 0.006, i, f"{v.mean():.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(methods)), [LABELS[m] for m in reversed(methods)], fontsize=8)
    ax.set_xlabel("Mean mIoU over held-out fires (95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIGS / "summary.pdf"); fig.savefig(FIGS / "summary.png", dpi=200)
    plt.close(fig)


def fig_train() -> None:
    hists = {m: json.loads((RESULTS / f"history_{m}.json").read_text())
             for m in ("unet_bands_idx", "unet_bands") if (RESULTS / f"history_{m}.json").exists()}
    if not hists:
        return
    fig, axes = plt.subplots(1, len(hists), figsize=(COL * len(hists) * 0.75, 1.5), squeeze=False, sharey=True)
    for ax, (m, h) in zip(axes[0], hists.items()):
        for fold, d in sorted(h.items()):
            l = np.convolve(d["loss"], np.ones(100) / 100, mode="valid")
            ax.plot(l, lw=1, color=C_UNET, alpha=0.35 + 0.13 * int(fold))
        ax.set_title(LABELS[m]); ax.set_xlabel("Training step")
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Training loss (100-step mean)")
    fig.tight_layout()
    fig.savefig(FIGS / "train.pdf"); fig.savefig(FIGS / "train.png", dpi=200)
    plt.close(fig)


def data_numbers() -> None:
    f = fires_table()
    macro("mFires", len(f)); macro("mFolds", f.fold.nunique())
    macro("mAcresM", f"{f.burnbndac.sum() / 1e6:.1f}")
    counts, subs = np.zeros(4), 0
    for n in f.fire:
        d = np.load(DATA / "fires" / f"{n}.npz")
        lab = d["label"]
        counts += np.bincount(lab[lab != IGNORE], minlength=4)
        subs += int(d["substituted"]) if "substituted" in d.files else 0
    macro("mPixelsM", f"{counts.sum() / 1e6:.1f}")
    for c, v in zip(CLASSES, counts / counts.sum()):
        macro(f"mShare{c.title()}", f"{100 * v:.1f}")
    macro("mSubstituted", subs)
    camp_label_agreement()
    macro("mYearMin", f.year.min()); macro("mYearMax", f.year.max())


def camp_label_agreement() -> None:
    """Agreement with the Camp Fire MTBS map of labels made by applying the generic
    dNBR thresholds, for two image sources (the MTBS scenes, and Landsat 8 median
    composites from gee/export_camp_fire.js) and two reflectance scalings."""
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject
    d = np.load(DATA / "fires" / "camp_2018.npz")
    lab = d["label"]
    k = lab != IGNORE
    nbr = lambda x: (x[3] - x[5]) / (x[3] + x[5] + 1e-6)

    def stretch(img):  # per-scene 2nd-98th percentile stretch of each band
        lo = np.nanpercentile(img, 2, axis=(1, 2))[:, None, None]
        hi = np.nanpercentile(img, 98, axis=(1, 2))[:, None, None]
        return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)

    def generic(a, b):
        x = (nbr(a) - nbr(b)) * 1000
        ok = k & np.isfinite(x)
        return np.digitize(x[ok], [100, 270, 440]), lab[ok]

    def agree(a, b):
        pred, ref = generic(a, b)
        return f"{100 * np.mean(pred == ref):.1f}"

    pre, post = d["pre"].astype(np.float32), d["post"].astype(np.float32)
    macro("mCampGenericPhys", agree(pre, post))
    macro("mCampStretch", agree(stretch(pre), stretch(post)))
    pred, ref = generic(pre, post)
    macro("mCampGenericHigh", f"{100 * np.mean(pred == 3):.0f}")
    macro("mCampMtbsHigh", f"{100 * np.mean(ref == 3):.0f}")

    imgs = []
    for name in ("pre", "post"):
        with rasterio.open(ROOT / "data" / "raw" / f"camp_fire_{name}_2018.tif") as src:
            raw = src.read().astype(np.float32) * 2.75e-05 - 0.2
            out = np.full((6,) + lab.shape, np.nan, np.float32)
            for c in range(6):
                reproject(raw[c], out[c], src_transform=src.transform, src_crs=src.crs,
                          dst_transform=Affine(*d["transform"]), dst_crs=str(d["crs"]),
                          resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        imgs.append(out)
    macro("mCampCompPhys", agree(*imgs))
    macro("mCampCompStretch", agree(stretch(imgs[0]), stretch(imgs[1])))


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    n = fires_table().shape[0]
    data_numbers()
    tables(df, n)
    paired(df)
    fig_thresholds()
    fig_per_fire(df)
    fig_maps(df)
    fig_train()
    fig_overview()
    fig_summary(df)
    lines = ["% Generated by robust/report.py. Do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (REPORT / "numbers.tex").write_text("\n".join(lines) + "\n")
    summary = df.groupby("method")[["miou", "macro_f1", "acc"]].mean().round(3).sort_values("miou")
    print(summary.to_string())
    print(f"wrote {len(macros)} macros")


if __name__ == "__main__":
    main()
