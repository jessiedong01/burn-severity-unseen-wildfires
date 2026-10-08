import numpy as np
import cv2
from skimage.filters import gaussian, threshold_multiotsu
from skimage.feature import canny
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from dnbr import CLASS_LABELS, CLASS_COLORS, dnbr_to_classes


def apply_gaussian(dnbr, sigma=1.5):
    return gaussian(dnbr, sigma=sigma)


def apply_canny(dnbr_smooth, sigma=1.0, low_threshold=0.05, high_threshold=0.15):
    d_min, d_max = dnbr_smooth.min(), dnbr_smooth.max()
    dnbr_norm = (dnbr_smooth - d_min) / (d_max - d_min + 1e-8)
    return canny(dnbr_norm, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)


def apply_otsu(dnbr_smooth):
    valid = dnbr_smooth[np.isfinite(dnbr_smooth)]
    thresholds = threshold_multiotsu(valid, classes=4)
    labels = np.digitize(dnbr_smooth, bins=thresholds).astype(np.int32)
    return labels, thresholds


def run_classical_pipeline(dnbr, sigma_gauss=1.5, sigma_canny=1.0):
    dnbr_smooth = apply_gaussian(dnbr, sigma=sigma_gauss)
    edges = apply_canny(dnbr_smooth, sigma=sigma_canny)
    pred_labels, thresholds = apply_otsu(dnbr_smooth)
    return dnbr_smooth, edges, pred_labels, thresholds


def compute_iou(pred, gt, num_classes=4):
    ious = {}
    for cls in range(num_classes):
        intersection = ((pred == cls) & (gt == cls)).sum()
        union = ((pred == cls) | (gt == cls)).sum()
        ious[CLASS_LABELS[cls]] = float(intersection) / float(union + 1e-8)
    ious["mIoU"] = np.mean(list(ious.values()))
    return ious


def compute_pixel_accuracy(pred, gt):
    return (pred == gt).mean()


def evaluate(pred_labels, gt_labels):
    ious = compute_iou(pred_labels, gt_labels)
    acc = compute_pixel_accuracy(pred_labels, gt_labels)
    print(f"  pixel accuracy: {acc:.4f}")
    for cls in CLASS_LABELS:
        print(f"  IoU [{cls}]: {ious[cls]:.4f}")
    print(f"  mIoU: {ious['mIoU']:.4f}")
    return ious, acc


def plot_classical_pipeline(fire_name, dnbr, dnbr_smooth, edges, pred_labels, gt_labels, ious, save_path=None):
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(
        f"{fire_name.replace('_', ' ').title()} — Classical Pipeline  (mIoU: {ious['mIoU']:.3f})",
        fontsize=13, fontweight="bold"
    )

    cmap_cls = mcolors.ListedColormap(CLASS_COLORS)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_cls.N)
    vmax = max(abs(np.nanmin(dnbr)), abs(np.nanmax(dnbr)))

    axes[0].imshow(dnbr, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Raw dNBR")
    axes[0].axis("off")

    axes[1].imshow(dnbr_smooth, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Gaussian Smoothed")
    axes[1].axis("off")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("Canny Edges")
    axes[2].axis("off")

    im = axes[3].imshow(pred_labels, cmap=cmap_cls, norm=norm)
    axes[3].set_title("Otsu Predicted")
    axes[3].axis("off")

    im2 = axes[4].imshow(gt_labels, cmap=cmap_cls, norm=norm)
    axes[4].set_title("USGS Ground Truth")
    axes[4].axis("off")

    cbar = plt.colorbar(im2, ax=axes[4], fraction=0.046, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(CLASS_LABELS)

    iou_text = "\n".join([f"{cls}: {ious[cls]:.3f}" for cls in CLASS_LABELS])
    fig.text(0.01, 0.01, f"Per-class IoU:\n{iou_text}", fontsize=8, va="bottom", family="monospace")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved {save_path}")
    plt.show()


def run_and_evaluate(fire_name, processed_dir="data/processed", output_dir="outputs"):
    from preprocess import load_tiff
    from dnbr import compute_dnbr, dnbr_to_classes

    proc_dir = Path(processed_dir) / fire_name
    pre_img, _ = load_tiff(proc_dir / "pre.tif")
    post_img, _ = load_tiff(proc_dir / "post.tif")

    dnbr, _, _ = compute_dnbr(pre_img, post_img)
    gt_labels = dnbr_to_classes(dnbr)

    dnbr_smooth, edges, pred_labels, thresholds = run_classical_pipeline(dnbr)
    print(f"otsu thresholds: {thresholds}")

    ious, acc = evaluate(pred_labels, gt_labels)

    out_dir = Path(output_dir) / fire_name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_classical_pipeline(
        fire_name, dnbr, dnbr_smooth, edges, pred_labels, gt_labels, ious,
        save_path=out_dir / "classical_pipeline.png"
    )
    return ious, acc


if __name__ == "__main__":
    import sys
    fire = sys.argv[1] if len(sys.argv) > 1 else "camp_fire"
    run_and_evaluate(fire)
