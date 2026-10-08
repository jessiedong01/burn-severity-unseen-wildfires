"""
Choose the fires for the study from the MTBS perimeter database.

Criteria: California or Oregon wildfires that started 2018-2021, mapped with an
MTBS extended assessment (post-fire image from the next growing season), at least
60,000 acres, with Landsat 8 or 9 pre- and post-fire scenes.

Fires that overlap are put in the same group, and whole groups are assigned to
cross-validation folds. A test fire therefore never shares ground with a
training fire, and no fire is thrown away.

Writes data/fires.csv.
"""
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent
PERIMS = ROOT / "data" / "mtbs" / "mtbs_perimeter_data" / "mtbs_perims_DD.shp"
ALBERS = "ESRI:102039"  # CRS of the MTBS severity mosaics
MAX_OVERLAP = 0.01      # shared area above this fraction of the smaller fire = same group
N_FOLDS = 5
YEARS = (2018, 2020, 2021)  # years whose MTBS mosaics we downloaded


def candidates() -> gpd.GeoDataFrame:
    g = gpd.read_file(PERIMS, engine="pyogrio")
    g["year"] = g["ig_date"].astype(str).str[:4].astype(int)
    keep = (
        g.event_id.str[:2].isin(["CA", "OR"])
        & g.year.isin(YEARS)
        & (g.incid_type == "Wildfire")
        & (g.asmnt_type == "Extended")
        & (g.burnbndac >= 60_000)
        & g.pre_id.str[0].isin(["8", "9"])
        & g.post_id.str[0].isin(["8", "9"])
    )
    return g[keep].to_crs(ALBERS).sort_values("burnbndac", ascending=False)


def overlap_groups(c: gpd.GeoDataFrame) -> list[int]:
    """Connected components of the 'overlaps' graph (union-find)."""
    geoms = list(c.geometry)
    parent = list(range(len(geoms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            inter = geoms[i].intersection(geoms[j]).area
            if inter > MAX_OVERLAP * min(geoms[i].area, geoms[j].area):
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(geoms))]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [ids[r] for r in roots]


def assign_folds(c: gpd.GeoDataFrame) -> list[int]:
    """Greedy balance of burned area across folds, largest group first."""
    area = c.groupby("group")["burnbndac"].sum().sort_values(ascending=False)
    load = [0.0] * N_FOLDS
    fold_of = {}
    for g, a in area.items():
        k = min(range(N_FOLDS), key=lambda i: load[i])
        fold_of[g] = k
        load[k] += a
    return [fold_of[g] for g in c["group"]]


def slug(name: str, year: int) -> str:
    return f"{name.lower().replace(' ', '_').replace('-', '_')}_{year}"


def main() -> None:
    s = candidates().reset_index(drop=True)
    print(f"{len(s)} candidate fires")
    s["group"] = overlap_groups(s)
    s["fold"] = assign_folds(s)
    s["fire"] = [slug(n, y) for n, y in zip(s.incid_name, s.year)]
    for g, members in s.groupby("group")["fire"]:
        if len(members) > 1:
            print(f"  overlap group {g}: {', '.join(members)}")
    cols = ["fire", "event_id", "incid_name", "year", "ig_date", "burnbndac", "pre_id", "post_id",
            "dnbr_offst", "low_t", "mod_t", "high_t", "group", "fold"]
    s[cols].to_csv(ROOT / "data" / "fires.csv", index=False)
    s[["fire", "geometry"]].to_file(ROOT / "data" / "fires.gpkg", driver="GPKG")
    print(f"{len(s)} fires -> data/fires.csv; burned acres per fold:", s.groupby("fold").burnbndac.sum().to_dict())
    print(s[cols].to_string(index=False))


if __name__ == "__main__":
    main()
