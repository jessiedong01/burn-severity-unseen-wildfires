"""
Download inputs and labels for every fire in data/fires.csv.

Inputs: the exact Landsat 8/9 Collection 2 Level-2 scenes MTBS used for the fire
(pre_id and post_id in the MTBS perimeter database), read from Microsoft
Planetary Computer and warped onto the 30 m grid of the MTBS severity mosaic.
Surface reflectance uses the Collection 2 scale factors.

Labels: the MTBS thematic burn-severity mosaic for the fire's year, read on its
own grid (never resampled), restricted to this fire's perimeter.

  MTBS code                         -> class
  1 unburned/underburned to low     -> 0 unburned
  2 low                             -> 1 low
  3 moderate                        -> 2 moderate
  4 high                            -> 3 high
  5 increased greenness             -> 0 unburned
  0 background, 6 non-processing    -> 255 ignore

Pixels outside the perimeter, or flagged as fill, cloud, dilated cloud, cirrus,
cloud shadow, or snow in either scene, are also set to 255.

Output: data/fires/<fire>.npz

Usage:
  python -m robust.fetch                 # all fires
  python -m robust.fetch delta_2018      # one fire
"""
import math
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "fires"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]  # OLI bands 2-7
BAND_NAMES = ["B2 blue", "B3 green", "B4 red", "B5 NIR", "B6 SWIR1", "B7 SWIR2"]
SR_SCALE, SR_OFFSET = 2.75e-05, -0.2
# QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow
QA_BAD = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 0}
IGNORE = 255
BUFFER_M = 1500
GRID_OFFSET = 15.0  # MTBS mosaic pixel edges sit at 30k + 15 m


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_scene(scene_id: str) -> dict:
    """MTBS scene ids look like 804403220180719: sensor, path, row, yyyymmdd."""
    return {"platform": f"landsat-{scene_id[0]}", "path": scene_id[1:4], "row": scene_id[4:7],
            "date": f"{scene_id[7:11]}-{scene_id[11:13]}-{scene_id[13:15]}"}


def find_item(catalog, scene_id: str, max_days: int = 16):
    """The exact scene MTBS used, or, if Planetary Computer lacks it, the closest
    Landsat 8/9 scene of the same path/row within max_days (logged)."""
    import datetime as dt
    s = parse_scene(scene_id)
    query = {"landsat:wrs_path": {"eq": s["path"]}, "landsat:wrs_row": {"eq": s["row"]}}
    items = list(catalog.search(collections=["landsat-c2-l2"], datetime=s["date"],
                                query={**query, "platform": {"eq": s["platform"]}}).items())
    if items:
        items.sort(key=lambda it: it.properties.get("landsat:collection_category", "") != "T1")
        return items[0], False
    day = dt.date.fromisoformat(s["date"])
    window = f"{day - dt.timedelta(days=max_days)}/{day + dt.timedelta(days=max_days)}"
    near = [it for it in catalog.search(collections=["landsat-c2-l2"], datetime=window, query=query).items()
            if it.properties.get("platform") in ("landsat-8", "landsat-9")
            and it.properties.get("eo:cloud_cover", 100) <= 20]
    if not near:
        raise LookupError(f"no Planetary Computer item within {max_days} days of scene {scene_id} ({s})")
    near.sort(key=lambda it: abs((it.datetime.date() - day).days))
    log(f"  scene {scene_id} missing on Planetary Computer; using {near[0].id}")
    return near[0], True


def snap_grid(bounds, res=30.0):
    xmin, ymin, xmax, ymax = bounds
    xmin = math.floor((xmin - GRID_OFFSET) / res) * res + GRID_OFFSET
    ymin = math.floor((ymin - GRID_OFFSET) / res) * res + GRID_OFFSET
    xmax = math.ceil((xmax - GRID_OFFSET) / res) * res + GRID_OFFSET
    ymax = math.ceil((ymax - GRID_OFFSET) / res) * res + GRID_OFFSET
    w, h = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
    return from_origin(xmin, ymax, res, res), w, h, (xmin, ymin, xmax, ymax)


def read_scene(item, crs, transform, w, h):
    """Return (6, h, w) float32 surface reflectance and a (h, w) bool validity mask."""
    sr = np.full((len(BANDS), h, w), np.nan, dtype=np.float32)
    for i, band in enumerate(BANDS):
        with rasterio.open(item.assets[band].href) as src, WarpedVRT(
            src, crs=crs, transform=transform, width=w, height=h,
            resampling=Resampling.bilinear, src_nodata=0, nodata=0,
        ) as vrt:
            dn = vrt.read(1).astype(np.float32)
        sr[i] = np.where(dn > 0, dn * SR_SCALE + SR_OFFSET, np.nan)
    with rasterio.open(item.assets["qa_pixel"].href) as src, WarpedVRT(
        src, crs=crs, transform=transform, width=w, height=h,
        resampling=Resampling.nearest, src_nodata=1, nodata=1,
    ) as vrt:
        qa = vrt.read(1)
    valid = ((qa & QA_BAD) == 0) & np.isfinite(sr).all(axis=0)
    return sr, valid


def fetch_fire(row, perims: gpd.GeoDataFrame, catalog) -> None:
    out = OUT / f"{row.fire}.npz"
    if out.exists():
        log(f"{row.fire}: exists, skipping")
        return
    mosaic = DATA / "mtbs" / f"mtbs_CONUS_{row.year}" / f"mtbs_CONUS_{row.year}.tif"
    with rasterio.open(mosaic) as m:
        crs = m.crs
        geom = perims.loc[perims.event_id == row.event_id].to_crs(crs).geometry.iloc[0]
        transform, w, h, bounds = snap_grid(geom.buffer(BUFFER_M).bounds)
        window = from_bounds(*bounds, transform=m.transform)
        codes = m.read(1, window=window.round_offsets().round_lengths(), boundless=True, fill_value=0)
    assert codes.shape == (h, w), (codes.shape, (h, w))
    log(f"{row.fire}: grid {w}x{h} px")

    (pre_item, pre_sub), (post_item, post_sub) = find_item(catalog, row.pre_id), find_item(catalog, row.post_id)
    pre, pre_ok = read_scene(pre_item, crs, transform, w, h)
    post, post_ok = read_scene(post_item, crs, transform, w, h)

    inside = geometry_mask([geom], out_shape=(h, w), transform=transform, invert=True)
    label = np.full((h, w), IGNORE, dtype=np.uint8)
    for code, cls in LABEL_MAP.items():
        label[codes == code] = cls
    label[~inside | ~pre_ok | ~post_ok] = IGNORE

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(out, pre=pre.astype(np.float16), post=post.astype(np.float16),
             pre_ok=pre_ok, post_ok=post_ok, inside=inside, label=label, mtbs_code=codes,
             transform=np.array(transform)[:6], crs=str(crs.to_wkt()),
             pre_item=pre_item.id, post_item=post_item.id, substituted=pre_sub or post_sub)
    n = (label != IGNORE).sum()
    shares = np.bincount(label[label != IGNORE], minlength=4) / max(n, 1)
    log(f"{row.fire}: {n:,} labeled px; unburned/low/mod/high = "
        + " / ".join(f"{s:.1%}" for s in shares)
        + f"; {pre_item.id} -> {post_item.id}")


def main() -> None:
    fires = pd.read_csv(DATA / "fires.csv", dtype={"pre_id": str, "post_id": str})
    if len(sys.argv) > 1:
        fires = fires[fires.fire.isin(sys.argv[1:])]
    perims = gpd.read_file(DATA / "mtbs" / "mtbs_perimeter_data" / "mtbs_perims_DD.shp", engine="pyogrio")
    perims = perims[perims.event_id.isin(fires.event_id)]
    catalog = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    for row in fires.itertuples():
        for attempt in range(3):
            try:
                fetch_fire(row, perims, catalog)
                break
            except Exception as e:  # network hiccups: retry, then move on
                log(f"{row.fire}: attempt {attempt + 1} failed: {e}")
                time.sleep(10)


if __name__ == "__main__":
    main()
