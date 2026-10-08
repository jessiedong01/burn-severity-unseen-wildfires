import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import rasterio

# USGS MTBS standard thresholds (https://www.mtbs.gov)
DNBR_THRESHOLDS = {
    "unburned": (-np.inf, 0.1),
    "low":      (0.1, 0.27),
    "moderate": (0.27, 0.44),
    "high":     (0.44, np.inf),
}
CLASS_LABELS = ["unburned", "low", "moderate", "high"]
CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"]


def compute_nbr(img):
    nir = img[:, :, 3].astype(np.float32)
    swir2 = img[:, :, 5].astype(np.float32)
    denom = nir + swir2
    return np.where(denom != 0, (nir - swir2) / denom, 0.0)


def compute_dnbr(pre_img, post_img):
    nbr_pre = compute_nbr(pre_img)
    nbr_post = compute_nbr(post_img)
    return nbr_pre - nbr_post, nbr_pre, nbr_post


def dnbr_to_classes(dnbr):
    labels = np.zeros(dnbr.shape, dtype=np.int32)
    for i, cls in enumerate(CLASS_LABELS):
        lo, hi = DNBR_THRESHOLDS[cls]
        labels[(dnbr > lo) & (dnbr <= hi)] = i
    return labels


def otsu_threshold_dnbr(dnbr):
    from skimage.filters import threshold_multiotsu
    valid = dnbr[np.isfinite(dnbr)]
    thresholds = threshold_multiotsu(valid, classes=4)
    labels = np.digitize(dnbr, bins=thresholds).astype(np.int32)
    return labels, thresholds


def compute_and_save(fire_name, processed_dir="data/processed", output_dir="outputs"):
    from preprocess import load_tiff

    proc_dir = Path(processed_dir) / fire_name
    pre_img, profile = load_tiff(proc_dir / "pre.tif")
    post_img, _ = load_tiff(proc_dir / "post.tif")

    dnbr, _, _ = compute_dnbr(pre_img, post_img)
    labels_usgs = dnbr_to_classes(dnbr)
    labels_otsu, otsu_thresholds = otsu_threshold_dnbr(dnbr)

    out_dir = Path(output_dir) / fire_name
    out_dir.mkdir(parents=True, exist_ok=True)

    single_band_profile = profile.copy()
    single_band_profile.update(count=1, dtype=rasterio.float32)
    with rasterio.open(out_dir / "dnbr.tif", "w", **single_band_profile) as dst:
        dst.write(dnbr[np.newaxis, :, :])

    label_profile = profile.copy()
    label_profile.update(count=1, dtype=rasterio.int32)
    with rasterio.open(out_dir / "labels_usgs.tif", "w", **label_profile) as dst:
        dst.write(labels_usgs[np.newaxis, :, :])
    with rasterio.open(out_dir / "labels_otsu.tif", "w", **label_profile) as dst:
        dst.write(labels_otsu[np.newaxis, :, :])

    print(f"{fire_name}: dNBR range [{np.nanmin(dnbr):.3f}, {np.nanmax(dnbr):.3f}]")
    print(f"  otsu thresholds: {otsu_thresholds}")
    for i, cls in enumerate(CLASS_LABELS):
        print(f"  {cls}: {(labels_usgs == i).mean() * 100:.1f}%")

    return dnbr, labels_usgs, labels_otsu


def plot_dnbr_and_labels(fire_name, pre_img, post_img, dnbr, labels, title_suffix="", save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"{fire_name.replace('_', ' ').title()} — {title_suffix}", fontsize=14, fontweight="bold")

    rgb_pre = np.stack([pre_img[:, :, 2], pre_img[:, :, 1], pre_img[:, :, 0]], axis=2)
    axes[0, 0].imshow(np.clip(rgb_pre, 0, 1))
    axes[0, 0].set_title("Pre-fire RGB")
    axes[0, 0].axis("off")

    rgb_post = np.stack([post_img[:, :, 2], post_img[:, :, 1], post_img[:, :, 0]], axis=2)
    axes[0, 1].imshow(np.clip(rgb_post, 0, 1))
    axes[0, 1].set_title("Post-fire RGB")
    axes[0, 1].axis("off")

    vmax = max(abs(np.nanmin(dnbr)), abs(np.nanmax(dnbr)))
    im = axes[0, 2].imshow(dnbr, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title("dNBR")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    cmap = mcolors.ListedColormap(CLASS_COLORS)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im2 = axes[1, 0].imshow(labels, cmap=cmap, norm=norm)
    axes[1, 0].set_title("Burn Severity Classes")
    axes[1, 0].axis("off")
    cbar = plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(CLASS_LABELS)

    counts = [(labels == i).sum() for i in range(4)]
    axes[1, 1].bar(CLASS_LABELS, counts, color=CLASS_COLORS, edgecolor="black")
    axes[1, 1].set_title("Class Distribution")
    axes[1, 1].set_ylabel("Pixel count")
    for i, c in enumerate(counts):
        axes[1, 1].text(i, c, f"{c/labels.size*100:.1f}%", ha="center", va="bottom", fontsize=9)

    axes[1, 2].hist(dnbr.ravel(), bins=100, color="steelblue", alpha=0.7, edgecolor="none")
    for thresh, color in zip([0.1, 0.27, 0.44], ["green", "orange", "red"]):
        axes[1, 2].axvline(thresh, color=color, linestyle="--", linewidth=1.5, label=f"{thresh}")
    axes[1, 2].set_title("dNBR Distribution")
    axes[1, 2].set_xlabel("dNBR")
    axes[1, 2].legend(title="USGS thresholds", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved {save_path}")
    plt.show()


if __name__ == "__main__":
    import sys
    fire = sys.argv[1] if len(sys.argv) > 1 else "camp_fire"
    compute_and_save(fire)
