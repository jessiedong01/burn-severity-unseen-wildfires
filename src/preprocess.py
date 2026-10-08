import numpy as np
import cv2
import rasterio
from pathlib import Path

# fires to process: name, year, pre start/end, post start/end, lon, lat, buffer_km
FIRES = [
    ("camp_fire",      2018, "2018-08-01", "2018-11-07", "2018-11-09", "2019-01-01", -121.57, 39.81, 25),
    ("dixie_fire",     2021, "2021-05-01", "2021-07-13", "2021-07-15", "2021-11-01", -121.15, 40.00, 40),
    ("caldor_fire",    2021, "2021-06-01", "2021-08-14", "2021-08-16", "2021-11-01", -120.40, 38.65, 30),
    ("bootleg_fire",   2021, "2021-05-01", "2021-07-06", "2021-07-08", "2021-10-01", -121.10, 42.40, 35),
    ("august_complex", 2020, "2020-06-01", "2020-08-16", "2020-08-18", "2020-12-01", -122.80, 39.80, 50),
]

BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]


def init_gee():
    import ee
    ee.Initialize()


def get_landsat_image(lon, lat, buffer_km, date_start, date_end):
    import ee
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_km * 1000).bounds()

    def mask_clouds(image):
        qa = image.select("QA_PIXEL")
        mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return image.updateMask(mask)

    img = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(region)
        .filterDate(date_start, date_end)
        .map(mask_clouds)
        .select(BANDS)
        .median()
        .clip(region)
    )
    return img, region


def export_fire_pair(fire_name, year, pre_start, pre_end, post_start, post_end, lon, lat, buffer_km, output_dir):
    import ee
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pre_img, region = get_landsat_image(lon, lat, buffer_km, pre_start, pre_end)
    post_img, _ = get_landsat_image(lon, lat, buffer_km, post_start, post_end)

    for label, img in [("pre", pre_img), ("post", post_img)]:
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=f"{fire_name}_{label}_{year}",
            folder="burn_severity",
            fileNamePrefix=f"{fire_name}_{label}_{year}",
            region=region,
            scale=30,
            crs="EPSG:4326",
            maxPixels=1e9,
        )
        task.start()
        print(f"started export: {fire_name} {label}")


def export_all_fires(output_dir="data/raw"):
    init_gee()
    for fire in FIRES:
        export_fire_pair(*fire, output_dir=output_dir)


def load_tiff(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile
    return np.transpose(data, (1, 2, 0)), profile


def normalize_band(band):
    b_min, b_max = np.nanpercentile(band, 2), np.nanpercentile(band, 98)
    return np.clip((band - b_min) / (b_max - b_min + 1e-8), 0, 1)


def to_uint8(band_norm):
    return (band_norm * 255).astype(np.uint8)


def sift_align(pre_img, post_img):
    pre_gray = to_uint8(normalize_band(pre_img[:, :, 2]))
    post_gray = to_uint8(normalize_band(post_img[:, :, 2]))

    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(pre_gray, None)
    kp2, des2 = sift.detectAndCompute(post_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

    print(f"  SIFT: {len(kp1)}/{len(kp2)} keypoints, {len(good)} good matches")

    if len(good) < 10:
        print("  too few matches, skipping alignment")
        return post_img

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    print(f"  homography inliers: {int(mask.sum())}/{len(good)}")

    h, w = pre_img.shape[:2]
    aligned = np.stack([
        cv2.warpPerspective(post_img[:, :, c], H, (w, h))
        for c in range(post_img.shape[2])
    ], axis=2)

    return aligned


def normalize_pair(pre_img, post_img):
    pre_norm = np.stack([normalize_band(pre_img[:, :, c]) for c in range(pre_img.shape[2])], axis=2)
    post_norm = np.stack([normalize_band(post_img[:, :, c]) for c in range(post_img.shape[2])], axis=2)
    return pre_norm, post_norm


def save_tiff(array, profile, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=array.shape[2], dtype=rasterio.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.transpose(array, (2, 0, 1)))


def preprocess_fire(fire_name, year, raw_dir="data/raw", processed_dir="data/processed"):
    raw_dir = Path(raw_dir)
    pre_path = raw_dir / f"{fire_name}_pre_{year}.tif"
    post_path = raw_dir / f"{fire_name}_post_{year}.tif"

    print(f"processing {fire_name} {year}...")
    pre_img, profile = load_tiff(pre_path)
    post_img, _ = load_tiff(post_path)

    post_aligned = sift_align(pre_img, post_img)
    pre_norm, post_norm = normalize_pair(pre_img, post_aligned)

    out_dir = Path(processed_dir) / fire_name
    save_tiff(pre_norm, profile, out_dir / "pre.tif")
    save_tiff(post_norm, profile, out_dir / "post.tif")
    print(f"  saved to {out_dir}")

    return pre_norm, post_norm, profile


if __name__ == "__main__":
    export_all_fires()
