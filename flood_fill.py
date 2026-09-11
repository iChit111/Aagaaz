"""'Bathtub fill' street-level flood extent from a surcharging SWMM node.

Grows a flood pool outward from a node's DEM pour point (terrain.py),
always adding the next-lowest neighboring cell first (a priority-flood /
Prim's-style fill), until the pooled volume matches the node's surcharge
volume. The pooled cells are then matched against nearby road geometry
to get a per-road depth.

This replaces main.py's old flat-average hack, which spread a node's
surcharge volume evenly across a fixed, hand-picked cluster of 15 roads
regardless of terrain. It's still a simplified planar/static-water-surface
approximation, not a real 2D shallow-water solve -- see the project's
TUFLOW/HEC-RAS 2D notes for what a production version would use instead.
"""

from __future__ import annotations

import heapq
import json
import math
import os

import numpy as np
import rasterio

from dem import DEM_PATH
from terrain import CATCHMENTS_PATH, load_catchments

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROADS_PATH = os.path.join(BASE_DIR, "frontend", "src", "pune_roads.json")

CELL_AREA_M2 = 900.0  # SRTM cells are ~30m x 30m
CELL_SIZE_M = 30.0
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# How far to search for the basin's true low point before flood-filling --
# matches terrain.py's MAX_CATCHMENT_RADIUS_M: a street inlet's flood pools
# locally, it doesn't chase a regional drainage minimum km away.
PIT_SEARCH_RADIUS_M = 200.0

CellIndex = tuple[int, int]  # (row, col)


def _local_candidates(
    elevation: np.ndarray, center_row: int, center_col: int, radius_m: float = PIT_SEARCH_RADIUS_M
) -> list[tuple[float, CellIndex]]:
    """All cells within radius_m of (center_row, center_col), sorted by
    elevation ascending: (elevation, (row, col)).

    This project tried growing the pool by discovering neighbors on the
    fly (a min-heap graph traversal, priority-flood style). That kept
    breaking: raw SRTM has flat plateaus at integer-meter steps, so a
    genuinely lower, connected cell is often only discovered *after* a
    same-or-higher plateau cell next to it has already been popped and
    "locked in" a water level -- and by the time the lower cell surfaces,
    either the non-decreasing-elevation assumption the volume math relies
    on is violated, or the newly-revealed low cell forces a big, late
    jump in achieved volume with no way to interpolate around it, since
    the level was already committed.

    Sidestepping that: since catchments are already capped to a local
    radius (terrain.py's MAX_CATCHMENT_RADIUS_M -- a street inlet floods
    its block, not a regional watershed), just enumerate every cell in
    that same local radius upfront and sort once. Filling in that fixed,
    complete order is correct by construction -- there is no "later
    discovery" left to invalidate an already-decided water level.
    """
    n_rows, n_cols = elevation.shape
    cell_radius = int(math.ceil(radius_m / CELL_SIZE_M))
    candidates = []
    for dr in range(-cell_radius, cell_radius + 1):
        for dc in range(-cell_radius, cell_radius + 1):
            r, c = center_row + dr, center_col + dc
            if not (0 <= r < n_rows and 0 <= c < n_cols):
                continue
            if math.hypot(dr * CELL_SIZE_M, dc * CELL_SIZE_M) > radius_m:
                continue
            candidates.append((float(elevation[r, c]), (r, c)))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _flood_fill_to_volume(
    elevation: np.ndarray, seed_row: int, seed_col: int, target_volume_m3: float
) -> tuple[list[CellIndex], float]:
    """Fill the local terrain around (seed_row, seed_col) until pooled
    volume reaches target_volume_m3.

    Works through _local_candidates' fixed, pre-sorted cell list from
    lowest to highest. At 30m/900m^2 SRTM cells, a single cell filled to
    just 1m deep already holds 900m^3 -- more than most of this project's
    surcharge volumes on its own -- so rather than stopping at whichever
    whole cell first meets or exceeds the target (overshooting by up to a
    full cell), this solves for the exact partial water level within the
    already-filled set before adding the next (necessarily higher, since
    the list is sorted) candidate: nothing else can get wet before the
    water level reaches that candidate's elevation, so volume is a known
    linear function of level in that range.

    Returns (filled_cells, water_level_m).
    """
    candidates = _local_candidates(elevation, seed_row, seed_col)

    filled: list[CellIndex] = []
    elev_sum = 0.0
    water_level = candidates[0][0] if candidates else float(elevation[seed_row, seed_col])

    for elev, (r, c) in candidates:
        n = len(filled)
        if n > 0:
            # Level needed to hit the target using only the already-filled
            # cells, i.e. without this next (higher) candidate joining the pool.
            candidate_level = target_volume_m3 / (CELL_AREA_M2 * n) + elev_sum / n
            if candidate_level <= elev:
                return filled, candidate_level

        filled.append((r, c))
        elev_sum += elev
        water_level = elev

        volume = CELL_AREA_M2 * (water_level * len(filled) - elev_sum)
        if volume >= target_volume_m3:
            return filled, water_level

    return filled, water_level


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def _densify(coords: list[list[float]], max_spacing_m: float = 15.0) -> list[tuple[float, float]]:
    """Interpolate extra points so consecutive samples are <= max_spacing_m
    apart. OSM way vertices are sparse on long straight segments -- a road
    can pass straight through a DEM cell without any of its original
    vertices landing inside it, at roughly half a 30m SRTM cell spacing."""
    if not coords:
        return []
    points = [tuple(coords[0])]
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        dist_m = _haversine_m(lon1, lat1, lon2, lat2)
        steps = max(1, math.ceil(dist_m / max_spacing_m))
        for i in range(1, steps + 1):
            t = i / steps
            points.append((lon1 + (lon2 - lon1) * t, lat1 + (lat2 - lat1) * t))
    return points


def _load_road_cell_indices(roads_path: str, transform) -> dict[str, list[CellIndex]]:
    """Return {road_id: [(row, col), ...]} for densified samples along each road."""
    with open(roads_path) as f:
        data = json.load(f)

    inverse = ~transform

    road_cells: dict[str, list[CellIndex]] = {}
    for feature in data.get("features", []):
        road_id = feature.get("id") or feature.get("properties", {}).get("id")
        coords = feature.get("geometry", {}).get("coordinates", [])
        if not road_id or not coords:
            continue
        indices = []
        for lon, lat in _densify(coords):
            col, row = inverse * (lon, lat)
            indices.append((int(round(row)), int(round(col))))
        road_cells[road_id] = indices
    return road_cells


class FloodExtentEstimator:
    """Caches the DEM and per-road cell indices, then answers per-node
    flood-extent and per-road depth queries cheaply."""

    def __init__(
        self,
        dem_path: str = DEM_PATH,
        catchments_path: str = CATCHMENTS_PATH,
        roads_path: str = ROADS_PATH,
    ):
        with rasterio.open(dem_path) as src:
            self._elevation = src.read(1).astype(float)
            transform = src.transform

        # _flood_fill_to_volume searches the full local radius around this
        # center and starts from whatever cell is lowest, so the pour
        # point just needs to be roughly in the right neighborhood.
        self._seed_points: dict[str, CellIndex] = {
            node_id: (info["pour_row"], info["pour_col"])
            for node_id, info in load_catchments(catchments_path).items()
        }

        n_rows, n_cols = self._elevation.shape
        road_cells = _load_road_cell_indices(roads_path, transform)
        self._road_cell_indices = {
            road_id: [(r, c) for r, c in cells if 0 <= r < n_rows and 0 <= c < n_cols]
            for road_id, cells in road_cells.items()
        }

    def flood_extent_by_index(
        self, node_id: str, surcharge_volume_m3: float
    ) -> dict[CellIndex, float]:
        """Return {(row, col): depth_cm} for the cells a node's surcharge pools into."""
        if surcharge_volume_m3 <= 0 or node_id not in self._seed_points:
            return {}

        seed_row, seed_col = self._seed_points[node_id]
        filled, water_level = _flood_fill_to_volume(
            self._elevation, seed_row, seed_col, surcharge_volume_m3
        )

        depths: dict[CellIndex, float] = {}
        for r, c in filled:
            depth_m = water_level - self._elevation[r, c]
            if depth_m > 0:
                depths[(r, c)] = depth_m * 100
        return depths

    def road_depths(self, cell_depth_cm: dict[CellIndex, float]) -> dict[str, float]:
        """Match pooled cells against road vertices; a road's depth is the
        deepest pooled cell any of its vertices falls in."""
        depths: dict[str, float] = {}
        for road_id, indices in self._road_cell_indices.items():
            matched = [cell_depth_cm[idx] for idx in indices if idx in cell_depth_cm]
            if matched:
                depths[road_id] = round(max(matched), 2)
        return depths


if __name__ == "__main__":
    estimator = FloodExtentEstimator()
    for node_id, volume in (("MANHOLE-PUNE-001", 112.45), ("MANHOLE-PUNE-001", 453.59)):
        extent = estimator.flood_extent_by_index(node_id, volume)
        roads = estimator.road_depths(extent)
        print(f"{node_id} @ {volume}m^3: {len(extent)} pooled cells, {len(roads)} roads affected")
        for road_id, depth in sorted(roads.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {road_id}: {depth}cm")
