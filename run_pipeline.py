import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

FIRE = sys.argv[1] if len(sys.argv) > 1 else "camp_fire"
YEAR = 2018
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
OUT_DIR = Path("outputs") / FIRE

OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocess import load_tiff, sift_align, normalize_pair, save_tiff

pre_raw_path = RAW_DIR / f"{FIRE}_pre_{YEAR}.tif"
post_raw_path = RAW_DIR / f"{FIRE}_post_{YEAR}.tif"

if not pre_raw_path.exists() or not post_raw_path.exists():
    print(f"missing tiffs in data/raw/ — download from GEE first")
    sys.exit(1)

print("loading images...")
pre_img, profile = load_tiff(pre_raw_path)
post_img, _ = load_tiff(post_raw_path)
print(f"  shape: {pre_img.shape}")

print("aligning with SIFT...")
post_aligned = sift_align(pre_img, post_img)

print("normalizing bands...")
pre_norm, post_norm = normalize_pair(pre_img, post_aligned)

proc_fire_dir = PROC_DIR / FIRE
proc_fire_dir.mkdir(parents=True, exist_ok=True)
save_tiff(pre_norm, profile, proc_fire_dir / "pre.tif")
save_tiff(post_norm, profile, proc_fire_dir / "post.tif")
print(f"saved processed pair to {proc_fire_dir}/")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from dnbr import compute_dnbr, dnbr_to_classes, otsu_threshold_dnbr, CLASS_LABELS, CLASS_COLORS, plot_dnbr_and_labels

print("\ncomputing dNBR...")
dnbr, nbr_pre, nbr_post = compute_dnbr(pre_norm, post_norm)
labels_usgs = dnbr_to_classes(dnbr)
labels_otsu, otsu_thresh = otsu_threshold_dnbr(dnbr)

print(f"dNBR range: [{np.nanmin(dnbr):.3f}, {np.nanmax(dnbr):.3f}]")
print(f"otsu thresholds: {otsu_thresh}")
for i, cls in enumerate(CLASS_LABELS):
    print(f"  {cls}: {(labels_usgs == i).mean() * 100:.1f}%")

single_profile = profile.copy()
single_profile.update(count=1, dtype=rasterio.float32)
with rasterio.open(OUT_DIR / "dnbr.tif", "w", **single_profile) as dst:
    dst.write(dnbr[np.newaxis])

label_profile = profile.copy()
label_profile.update(count=1, dtype=rasterio.int32)
with rasterio.open(OUT_DIR / "labels_usgs.tif", "w", **label_profile) as dst:
    dst.write(labels_usgs[np.newaxis])

plot_dnbr_and_labels(
    FIRE, pre_norm, post_norm, dnbr, labels_usgs,
    title_suffix="dNBR + USGS Labels",
    save_path=str(OUT_DIR / "dnbr_overview.png"),
)

from classical import run_classical_pipeline, evaluate, plot_classical_pipeline

print("\nrunning classical pipeline...")
dnbr_smooth, edges, pred_labels, thresholds = run_classical_pipeline(dnbr)
print(f"otsu thresholds on smoothed dNBR: {thresholds}")
ious, acc = evaluate(pred_labels, labels_usgs)

plot_classical_pipeline(
    FIRE, dnbr, dnbr_smooth, edges, pred_labels, labels_usgs, ious,
    save_path=str(OUT_DIR / "classical_pipeline.png"),
)

print("\nsaving extra figures...")

fig, ax = plt.subplots(figsize=(7, 6))
cmap_cls = mcolors.ListedColormap(CLASS_COLORS)
norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_cls.N)
sample = np.random.default_rng(42).choice(nbr_pre.size, size=min(50_000, nbr_pre.size), replace=False)
sc = ax.scatter(
    nbr_pre.ravel()[sample], nbr_post.ravel()[sample],
    c=labels_usgs.ravel()[sample], cmap=cmap_cls, norm=norm, s=1, alpha=0.4,
)
ax.set_xlabel("NBR pre-fire")
ax.set_ylabel("NBR post-fire")
ax.set_title(f"{FIRE.replace('_', ' ').title()} — NBR scatter")
cbar = plt.colorbar(sc, ax=ax, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(CLASS_LABELS)
fig.tight_layout()
fig.savefig(OUT_DIR / "nbr_scatter.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3))
ax.axis("off")
rows = [[cls, f"{ious[cls]:.4f}"] for cls in CLASS_LABELS]
rows += [["mIoU", f"{ious['mIoU']:.4f}"], ["Pixel Acc", f"{acc:.4f}"]]
tbl = ax.table(cellText=rows, colLabels=["Class", "IoU"], loc="center", cellLoc="center")
tbl.scale(1, 1.5)
ax.set_title("Classical Pipeline — Evaluation Metrics", pad=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "metrics_table.png", dpi=150)
plt.close(fig)

print(f"\ndone — outputs saved to {OUT_DIR}/")
