# Wildfire Burn Severity Mapping via Satellite Imagery

How closely can an automatic method reproduce USGS MTBS burn severity maps
on a fire it has never seen?

The study (`robust/`, paper in `paper/main.pdf`) scores every method against
MTBS analyst maps on 31 California and Oregon wildfires (2018–2021), with
five-fold cross-validation that holds out whole fires. The first version of the
pipeline is kept in `src/` and `notebooks/`; Section 8 of the paper explains
what its evaluation measured and why the numbers changed.

## Pipeline (`robust/`)

| Step | Script |
|------|--------|
| Pick fires and folds from the MTBS perimeter database | `fires.py` |
| Download the exact Landsat scenes MTBS used (Planetary Computer) and MTBS labels | `fetch.py` |
| Index baselines, per-pixel gradient boosting, analyst-threshold oracle | `baselines.py` |
| U-Net per fold (12 bands, or 12 bands + 5 indices) | `unet.py` |
| Tables, figures, and every number in the report | `report.py` |
| Everything, in order | `run_all.sh` |

## Data

- Labels: MTBS thematic burn severity mosaics (ScienceBase) and perimeters
  (`data/mtbs/`, not committed; see `fetch.py` for sources).
- Inputs: Landsat 8/9 Collection 2 Level-2 surface reflectance, read from
  Microsoft Planetary Computer and warped onto the MTBS 30 m grid.

## Results (mean mIoU over 31 held-out fires)

| Method | mIoU |
|--------|------|
| Gaussian + multi-Otsu (original pipeline) | 0.456 |
| dNBR, generic thresholds | 0.564 |
| dNBR, thresholds learned on training fires | 0.684 |
| U-Net, 12 bands + indices | 0.686 |
| Oracle: each fire's own analyst thresholds | 0.735 |

See `paper/main.pdf` for the full table, confidence intervals, and figures.
