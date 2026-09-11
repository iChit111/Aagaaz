"""Hydrological conditioning and catchment delineation over the Pune DEM.

Conditions the raw SRTM tile (fills pits/depressions, resolves flats --
raw SRTM has noise that creates fake sinks), computes D8 flow direction
and accumulation, then delineates the contributing catchment for each
SWMM junction. Each catchment mask is what Phase D (rainfall -> inflow)
and Phase E (bathtub-fill depth) will aggregate over.

Note: the DEM stays in geographic coordinates (EPSG:4326, degrees) rather
than a projected CRS. That makes cell "distances" in meters inaccurate,
but catchment membership only depends on relative elevation and D8
direction, not absolute distance -- fine for demo-grade delineation, not
for anything that needs real slope/velocity.
"""

from __future__ import annotations

import math
import os

import numpy as np
import rasterio

# pysheds 0.5 calls np.in1d, removed in numpy>=2.0 in favor of np.isin
# (identical semantics for pysheds' usage: no positional-only kwargs relied on).
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from pysheds.grid import Grid

from dem import DEM_PATH, fetch_dem
from network_topology import JUNCTION_COORDINATES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CATCHMENTS_PATH = os.path.join(DATA_DIR, "catchments.npz")

# D8 direction encoding pysheds expects (ESRI convention: N, NE, E, ... clockwise).
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)

# A cell counts as a "drain-worthy" flow path once this many upstream cells
# feed it -- used to snap a junction's raw coordinate onto the nearest
# actual flow channel before delineating its catchment.
ACCUMULATION_THRESHOLD = 5

# Full D8 watershed delineation can snap a junction onto a major regional
# flow path and hand back a catchment covering a large fraction of the
# whole DEM tile (observed: one junction's true watershed was ~5,000 cells,
# ~500x its neighbor's) -- appropriate for a natural stream gauge, not a
# street inlet, which is engineered to intercept only its local block.
# Capping to a radius around the pour point keeps catchments street-scale.
MAX_CATCHMENT_RADIUS_M = 200.0


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def condition_and_flow(dem_path: str = DEM_PATH):
    """Load the DEM, hydrologically condition it, and return flow direction/accumulation."""
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    conditioned = grid.resolve_flats(flooded)

    fdir = grid.flowdir(conditioned, dirmap=DIRMAP)
    acc = grid.accumulation(fdir, dirmap=DIRMAP)

    return grid, conditioned, fdir, acc


def delineate_catchments(
    junction_coordinates: dict[str, tuple[float, float]] = JUNCTION_COORDINATES,
    dem_path: str = DEM_PATH,
) -> dict[str, dict]:
    """Return {node_id: {"mask": bool catchment array, "pour_row": int, "pour_col": int}}.

    pour_row/pour_col is the junction's coordinate snapped onto the DEM's
    nearest actual flow channel -- both the catchment's outlet and the seed
    cell Phase E's bathtub-fill grows a flood pool from.
    """
    grid, _conditioned, fdir, acc = condition_and_flow(dem_path)

    with rasterio.open(dem_path) as src:
        transform = src.transform

    catchments: dict[str, dict] = {}
    for node_id, (lon, lat) in junction_coordinates.items():
        snapped_x, snapped_y = grid.snap_to_mask(acc > ACCUMULATION_THRESHOLD, (lon, lat))
        catch = grid.catchment(
            x=snapped_x, y=snapped_y, fdir=fdir, dirmap=DIRMAP, xytype="coordinate"
        )
        pour_row, pour_col = rasterio.transform.rowcol(transform, snapped_x, snapped_y)

        mask = np.asarray(catch, dtype=bool)
        rows, cols = np.where(mask)
        for r, c in zip(rows, cols):
            cell_lon, cell_lat = transform * (c + 0.5, r + 0.5)
            if _haversine_m(cell_lon, cell_lat, snapped_x, snapped_y) > MAX_CATCHMENT_RADIUS_M:
                mask[r, c] = False

        catchments[node_id] = {
            "mask": mask,
            "pour_row": int(pour_row),
            "pour_col": int(pour_col),
        }

    return catchments


def save_catchments(catchments: dict[str, dict], dest_path: str = CATCHMENTS_PATH) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {}
    for node_id, info in catchments.items():
        payload[f"{node_id}__mask"] = info["mask"]
        payload[f"{node_id}__pour"] = np.array([info["pour_row"], info["pour_col"]])
    np.savez_compressed(dest_path, **payload)
    return dest_path


def load_catchments(path: str = CATCHMENTS_PATH) -> dict[str, dict]:
    with np.load(path) as data:
        node_ids = sorted({key.rsplit("__", 1)[0] for key in data.files})
        return {
            node_id: {
                "mask": data[f"{node_id}__mask"],
                "pour_row": int(data[f"{node_id}__pour"][0]),
                "pour_col": int(data[f"{node_id}__pour"][1]),
            }
            for node_id in node_ids
        }


if __name__ == "__main__":
    fetch_dem()
    catchments = delineate_catchments()
    for node_id, info in catchments.items():
        # SRTM cells are ~30m, so cell_count * 900 m^2 is a rough catchment area.
        cells = info["mask"].sum()
        print(
            f"{node_id}: {cells} cells (~{cells * 900:,.0f} m^2), "
            f"pour point at row={info['pour_row']} col={info['pour_col']}"
        )

    path = save_catchments(catchments)
    print(f"Catchment masks saved to {path}")
