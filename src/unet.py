"""
unet.py
-------
U-Net segmentation model for burn severity classification.
Input: stacked multispectral bands (B, 6, H, W) — RGB + NIR + SWIR1 + SWIR2
Output: (B, 4, H, W) class logits

Training loop and dataset loader.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm


# ── Model ──────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=4, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.head = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.head(x)


# ── Dataset ────────────────────────────────────────────────────────────────────

class WildfireDataset(Dataset):
    """
    Loads patches from preprocessed fire tiffs.
    Each item: (6, patch_size, patch_size) float32 tensor, (patch_size, patch_size) int64 label tensor
    """
    def __init__(self, fire_dirs, patch_size=256, stride=128, split="train", val_ratio=0.15):
        self.patches = []
        self.patch_size = patch_size

        for fire_dir in fire_dirs:
            fire_dir = Path(fire_dir)
            pre = self._load(fire_dir / "pre.tif")   # (H, W, 6)
            post = self._load(fire_dir / "post.tif")
            labels = self._load_labels(fire_dir / "../../outputs" / fire_dir.name / "labels_usgs.tif")

            # Stack all 6 pre bands + 6 post bands... or just post bands + dNBR?
            # Using post-fire bands directly as model input (shape H,W,6)
            H, W = post.shape[:2]

            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    img_patch = post[y:y+patch_size, x:x+patch_size, :]  # (P, P, 6)
                    lbl_patch = labels[y:y+patch_size, x:x+patch_size]   # (P, P)
                    self.patches.append((img_patch, lbl_patch))

        # Train/val split
        n = len(self.patches)
        split_idx = int(n * (1 - val_ratio))
        if split == "train":
            self.patches = self.patches[:split_idx]
        else:
            self.patches = self.patches[split_idx:]

    def _load(self, path):
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        return np.transpose(data, (1, 2, 0))

    def _load_labels(self, path):
        with rasterio.open(path) as src:
            return src.read(1).astype(np.int64)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img, lbl = self.patches[idx]
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # (6, P, P)
        lbl_t = torch.from_numpy(lbl)                    # (P, P)
        return img_t, lbl_t


# ── Training ───────────────────────────────────────────────────────────────────

def train(fire_dirs, epochs=20, lr=1e-3, batch_size=8, patch_size=256, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    train_ds = WildfireDataset(fire_dirs, patch_size=patch_size, split="train")
    val_ds   = WildfireDataset(fire_dirs, patch_size=patch_size, split="val")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNet(in_channels=6, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_miou": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, lbls in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)
            loss = criterion(preds, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_lbls = [], []
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, lbls).item()
                all_preds.append(preds.argmax(1).cpu().numpy())
                all_lbls.append(lbls.cpu().numpy())

        all_preds = np.concatenate([p.ravel() for p in all_preds])
        all_lbls  = np.concatenate([l.ravel() for l in all_lbls])

        # mIoU
        from classical import compute_iou
        ious = compute_iou(all_preds.reshape(-1, 1), all_lbls.reshape(-1, 1))

        tl = train_loss / len(train_dl)
        vl = val_loss / len(val_dl)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_miou"].append(ious["mIoU"])
        print(f"  train_loss={tl:.4f}  val_loss={vl:.4f}  val_mIoU={ious['mIoU']:.4f}")

    torch.save(model.state_dict(), "outputs/unet_weights.pt")
    return model, history


if __name__ == "__main__":
    from pathlib import Path
    fire_dirs = list(Path("data/processed").glob("*"))
    model, history = train(fire_dirs)
