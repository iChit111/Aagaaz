"""Converts the synthetic nowcast into per-node inflow for SWMM.

Aggregates rainfall intensity (nowcast.py) over each junction's
DEM-derived catchment (terrain.py) and converts it to a volumetric
inflow rate -- replacing the flat single-manhole, single-scalar rainfall
input main.py used to drive pyswmm's Simulation with.
"""

from __future__ import annotations

import os

import numpy as np
import rasterio

from dem import DEM_PATH
from nowcast import DEFAULT_STORM, StormCell, intensity_mm_per_hr
from terrain import CATCHMENTS_PATH, load_catchments

CELL_AREA_M2 = 900.0  # SRTM cells are ~30m x 30m
RUNOFF_COEFFICIENT = 0.9  # concrete-heavy urban catchment, matches main.py's prior assumption


def _catchment_cell_coordinates(
    catchments_path: str, dem_path: str
) -> dict[str, list[tuple[float, float]]]:
    """Return {node_id: [(lon, lat), ...]} for every cell in each catchment."""
    try:
        catchments = load_catchments(catchments_path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{catchments_path} not found. Run `python3 terrain.py` first to "
            "delineate catchments from the DEM."
        ) from exc

    with rasterio.open(dem_path) as src:
        transform = src.transform

    coords: dict[str, list[tuple[float, float]]] = {}
    for node_id, info in catchments.items():
        rows, cols = np.where(info["mask"])
        coords[node_id] = [transform * (col + 0.5, row + 0.5) for row, col in zip(rows, cols)]
    return coords


class CatchmentRunoff:
    """Precomputes catchment geometry once, then cheaply answers per-timestep inflow queries."""

    def __init__(self, catchments_path: str = CATCHMENTS_PATH, dem_path: str = DEM_PATH):
        self._cell_coords = _catchment_cell_coordinates(catchments_path, dem_path)
        self._area_m2 = {
            node_id: len(cells) * CELL_AREA_M2 for node_id, cells in self._cell_coords.items()
        }

    @property
    def node_ids(self) -> list[str]:
        return list(self._cell_coords)

    def catchment_area_m2(self, node_id: str) -> float:
        return self._area_m2.get(node_id, 0.0)

    def mean_rainfall_mm_hr(
        self, node_id: str, elapsed_min: float, storm: StormCell = DEFAULT_STORM
    ) -> float:
        cells = self._cell_coords.get(node_id, [])
        if not cells:
            return 0.0
        return sum(intensity_mm_per_hr(lon, lat, elapsed_min, storm) for lon, lat in cells) / len(
            cells
        )

    def inflow_cms(
        self, node_id: str, elapsed_min: float, storm: StormCell = DEFAULT_STORM
    ) -> float:
        """Volumetric inflow (m^3/s) into `node_id` from its catchment at a given time."""
        rainfall_mm_hr = self.mean_rainfall_mm_hr(node_id, elapsed_min, storm)
        rainfall_m_per_s = rainfall_mm_hr / 1000.0 / 3600.0
        return rainfall_m_per_s * self.catchment_area_m2(node_id) * RUNOFF_COEFFICIENT


if __name__ == "__main__":
    runoff = CatchmentRunoff()
    for node_id in runoff.node_ids:
        print(f"{node_id}: catchment area = {runoff.catchment_area_m2(node_id):,.0f} m^2")
        for t in (0, 90, 180):
            print(
                f"  t={t:>3}min  rainfall={runoff.mean_rainfall_mm_hr(node_id, t):6.2f}mm/hr"
                f"  inflow={runoff.inflow_cms(node_id, t):.5f} m^3/s"
            )
