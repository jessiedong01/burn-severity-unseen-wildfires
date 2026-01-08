"""
U-Net trained and tested with the fire-grouped 5-fold split.

For each fold the model trains on the other four folds' fires for a fixed number
of steps (no early stopping, so test fires never influence training) and then
predicts every test fire with an overlapping sliding window.

Variants:
  bands      12 channels: pre- and post-fire surface reflectance, bands 2-7
  bands_idx  17 channels: the 12 bands plus NBR pre, NBR post, dNBR, RdNBR, RBR

Usage:
  python -m robust.unet --variant bands_idx [--iters 3000] [--folds 0 1 2 3 4]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from robust.common import IGNORE, RESULTS, confusion, fires_table, indices, load_fire, scores

PATCH, BATCH = 256, 16
CLIP = None  # set by --clip; None trains exactly as the first pass did


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Model ────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin, ncls=4, feats=(32, 64, 128, 256)):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        c = cin
        for f in feats:
            self.downs.append(DoubleConv(c, f)); c = f
        self.bottleneck = DoubleConv(feats[-1], feats[-1] * 2)
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))
        self.head = nn.Conv2d(feats[0], ncls, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = self.ups[i + 1](torch.cat([skips[-1 - i // 2], x], dim=1))
        return self.head(x)


# ── Inputs ───────────────────────────────────────────────────────────────────

def build_input(f: dict, variant: str) -> np.ndarray:
    chans = [f["pre"][i] for i in range(6)] + [f["post"][i] for i in range(6)]
    if variant == "bands_idx":
        ix = indices(f["pre"], f["post"])
        chans += [ix["nbr_pre"], ix["nbr_post"], ix["dnbr"] / 1000,
                  np.clip(ix["rdnbr"] / 1000, -5, 5), ix["rbr"] / 1000]
    return np.stack(chans).astype(np.float32)  # (C, H, W), NaN where invalid


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(z, nan=0.0).astype(np.float16)


class PatchSampler:
    """Random 256x256 crops centred on labelled pixels. Fires are drawn with
    probability proportional to sqrt(labelled pixels), so huge fires do not
    swamp small ones."""

    def __init__(self, xs, ys, seed):
        self.xs, self.ys = xs, ys
        self.rng = np.random.default_rng(seed)
        self.labelled = [np.flatnonzero(y.ravel() != IGNORE) for y in ys]
        w = np.sqrt([len(l) for l in self.labelled]); self.p = w / w.sum()

    def batch(self, n):
        xb = np.zeros((n, self.xs[0].shape[0], PATCH, PATCH), np.float32)
        yb = np.full((n, PATCH, PATCH), IGNORE, np.int64)
        for b in range(n):
            k = self.rng.choice(len(self.xs), p=self.p)
            x, y = self.xs[k], self.ys[k]
            H, W = y.shape
            c = self.rng.choice(self.labelled[k]); cy, cx = divmod(int(c), W)
            y0 = int(np.clip(cy - PATCH // 2, 0, max(H - PATCH, 0)))
            x0 = int(np.clip(cx - PATCH // 2, 0, max(W - PATCH, 0)))
            xp, yp = x[:, y0:y0 + PATCH, x0:x0 + PATCH], y[y0:y0 + PATCH, x0:x0 + PATCH]
            h, w = yp.shape
            xb[b, :, :h, :w], yb[b, :h, :w] = xp, yp
            if self.rng.random() < 0.5:
                xb[b], yb[b] = xb[b][:, :, ::-1], yb[b][:, ::-1]
            if self.rng.random() < 0.5:
                xb[b], yb[b] = xb[b][:, ::-1, :], yb[b][::-1, :]
            r = self.rng.integers(4)
            xb[b], yb[b] = np.rot90(xb[b], r, axes=(1, 2)), np.rot90(yb[b], r)
        return torch.from_numpy(xb), torch.from_numpy(yb)


@torch.no_grad()
def predict(model, x: np.ndarray, device, stride=128) -> np.ndarray:
    """Sliding window with Hann weighting; reflect-pads so every pixel is covered."""
    C, H, W = x.shape
    ph, pw = max(PATCH, -(-H // stride) * stride + PATCH - stride), max(PATCH, -(-W // stride) * stride + PATCH - stride)
    xp = np.pad(x.astype(np.float32), ((0, 0), (0, ph - H), (0, pw - W)), mode="reflect")
    win = np.outer(np.hanning(PATCH + 2)[1:-1], np.hanning(PATCH + 2)[1:-1]).astype(np.float32)
    acc = np.zeros((4, ph, pw), np.float32); wsum = np.zeros((ph, pw), np.float32)
    coords = [(r, c) for r in range(0, ph - PATCH + 1, stride) for c in range(0, pw - PATCH + 1, stride)]
    model.eval()
    for i in range(0, len(coords), 32):
        chunk = coords[i:i + 32]
        xb = torch.from_numpy(np.stack([xp[:, r:r + PATCH, c:c + PATCH] for r, c in chunk])).to(device)
        prob = torch.softmax(model(xb), 1).float().cpu().numpy()
        for (r, c), p in zip(chunk, prob):
            acc[:, r:r + PATCH, c:c + PATCH] += p * win
            wsum[r:r + PATCH, c:c + PATCH] += win
    return (acc[:, :H, :W] / np.maximum(wsum[:H, :W], 1e-6)).argmax(0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["bands", "bands_idx"], default="bands_idx")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--clip", type=float, default=None, help="gradient-norm clipping (used for reruns)")
    args = ap.parse_args()
    global CLIP
    CLIP = args.clip
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    name = f"unet_{args.variant}"
    fires = fires_table()
    folds = args.folds if args.folds is not None else sorted(fires.fold.unique())

    raw = {}
    for n in fires.fire:
        f = load_fire(n)
        raw[n] = (build_input(f, args.variant), f["label"], f["valid"])
    log(f"loaded {len(raw)} fires, {raw[fires.fire[0]][0].shape[0]} channels, device {device}")

    rows, history = [], {}
    old_csv, old_hist = RESULTS / f"metrics_{name}.csv", RESULTS / f"history_{name}.json"
    if args.folds is not None and old_csv.exists():  # rerun of some folds: keep the others
        old = pd.read_csv(old_csv)
        rows = old[~old.fold.isin(folds)].to_dict("records")
        history = {int(k): v for k, v in json.loads(old_hist.read_text()).items() if int(k) not in folds}
    for fold in folds:
        train = fires[fires.fold != fold].fire.tolist()
        test = fires[fires.fold == fold].fire.tolist()
        # standardisation and class weights from training fires only
        sample = np.concatenate([raw[n][0][:, raw[n][2]][:, ::50] for n in train], axis=1)
        mean, std = np.nanmean(sample, 1), np.nanstd(sample, 1) + 1e-6
        counts = sum(np.bincount(raw[n][1][raw[n][1] != IGNORE], minlength=4) for n in train)
        weight = (counts / counts.sum()) ** -0.5; weight = weight / weight.mean()

        torch.manual_seed(fold)
        sampler = PatchSampler([standardize(raw[n][0], mean, std) for n in train],
                               [raw[n][1] for n in train], seed=fold)
        model = UNet(raw[train[0]][0].shape[0]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.iters, pct_start=0.05)
        lossf = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=device), ignore_index=IGNORE)
        losses, t0, skipped = [], time.time(), 0
        model.train()
        for it in range(args.iters):
            xb, yb = sampler.batch(BATCH)
            loss = lossf(model(xb.to(device)), yb.to(device))
            opt.zero_grad()
            if not torch.isfinite(loss):  # skip a bad batch instead of poisoning the weights
                skipped += 1; sched.step(); losses.append(float("nan")); continue
            loss.backward()
            if CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP)
            opt.step(); sched.step()
            losses.append(loss.item())
            if (it + 1) % 250 == 0:
                log(f"{name} fold {fold}: step {it + 1}/{args.iters} loss {np.mean(losses[-250:]):.4f} "
                    f"({(time.time() - t0) / (it + 1):.2f} s/step)")
        history[int(fold)] = {"loss": losses, "class_weight": weight.tolist(),
                              "train": train, "test": test, "clip": CLIP, "skipped": skipped}
        torch.save(model.state_dict(), RESULTS / f"{name}_fold{fold}.pt")

        out = RESULTS / "preds" / name
        out.mkdir(parents=True, exist_ok=True)
        for n in test:
            x, y, valid = raw[n]
            pred = predict(model, standardize(x, mean, std), device)
            pred[~valid] = IGNORE
            np.save(out / f"{n}.npy", pred)
            cm = confusion(np.where(pred == IGNORE, 0, pred), y)
            s = scores(cm)
            rows.append({"method": name, "fire": n, "fold": int(fold), **s, "cm": json.dumps(cm.tolist())})
            log(f"{name} fold {fold}: {n} mIoU {s['miou']:.3f}")
        pd.DataFrame(rows).to_csv(RESULTS / f"metrics_{name}.csv", index=False)
        (RESULTS / f"history_{name}.json").write_text(json.dumps(history))
    log("done")


if __name__ == "__main__":
    main()
