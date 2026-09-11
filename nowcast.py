"""Synthetic rainfall nowcast generator.

Real IMD Doppler Weather Radar feeds aren't publicly accessible, so this
stands in for a live nowcast: a spatially-localized storm cell with a
slow drift and a bell-shaped intensity profile over the 0-3hr window,
queryable at any (lon, lat, elapsed_min). It's a placeholder, not a
prediction -- swap this module out for a real radar-derived nowcast
without touching anything downstream, since callers only ever use
`intensity_mm_per_hr`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from network_topology import JUNCTION_COORDINATES

# Matches pune_roads.json's centroid, kept for reference/other callers --
# the default storm below centers on the SWMM junctions instead (see
# JUNCTION_CENTER), since that's the area the demo network can actually feel.
AOI_CENTER = (73.8414, 18.5158)


@dataclass(frozen=True)
class StormCell:
    center0: tuple[float, float]  # (lon, lat) at elapsed_min=0
    velocity_deg_per_min: tuple[float, float]  # slow drift (lon, lat) per minute
    peak_intensity_mm_hr: float
    radius_m: float  # spatial falloff (Gaussian std-dev, in meters)
    peak_at_min: float  # when in the 0-3hr window intensity peaks
    spread_min: float  # temporal falloff (Gaussian std-dev, in minutes)


# Centered on the demo network's junctions rather than the wider AOI, so
# the storm actually passes over the catchments the SWMM model can feel.
JUNCTION_CENTER = (
    sum(c[0] for c in JUNCTION_COORDINATES.values()) / len(JUNCTION_COORDINATES),
    sum(c[1] for c in JUNCTION_COORDINATES.values()) / len(JUNCTION_COORDINATES),
)

# A monsoon-scale urban cell: peaks at "Very Heavy Rain" (per the frontend's
# IMD intensity brackets), drifting gently rather than sweeping the whole
# AOI, since convective city storms are closer to quasi-stationary than
# fast-moving frontal systems. radius_m=1800 matches a realistic convective
# cell footprint (~3-5km across) -- narrower and the storm collapses to
# near-zero rainfall just a km from its center, which starved every
# catchment regardless of how high the intensity dial was turned up.
DEFAULT_STORM = StormCell(
    center0=JUNCTION_CENTER,
    velocity_deg_per_min=(0.00003, -0.00002),
    peak_intensity_mm_hr=70.0,
    radius_m=1800.0,
    peak_at_min=90.0,
    spread_min=60.0,
)


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def _temporal_envelope(storm: StormCell, elapsed_min: float) -> float:
    """Bell-shaped ramp: low at the edges of the window, peaks mid-storm."""
    return math.exp(-(((elapsed_min - storm.peak_at_min) / storm.spread_min) ** 2))


def storm_center_at(elapsed_min: float, storm: StormCell = DEFAULT_STORM) -> tuple[float, float]:
    return (
        storm.center0[0] + storm.velocity_deg_per_min[0] * elapsed_min,
        storm.center0[1] + storm.velocity_deg_per_min[1] * elapsed_min,
    )


def intensity_mm_per_hr(
    lon: float, lat: float, elapsed_min: float, storm: StormCell = DEFAULT_STORM
) -> float:
    """Rainfall intensity at a point and time, mm/hr."""
    cx, cy = storm_center_at(elapsed_min, storm)
    dist_m = _haversine_m(lon, lat, cx, cy)
    spatial_falloff = math.exp(-(dist_m**2) / (2 * storm.radius_m**2))
    return storm.peak_intensity_mm_hr * spatial_falloff * _temporal_envelope(storm, elapsed_min)


def sample_grid(
    bounds: dict,
    elapsed_min: float,
    resolution: int = 15,
    storm: StormCell = DEFAULT_STORM,
) -> dict:
    """Sample the storm field on a resolution x resolution grid over `bounds`
    ({south, north, west, east}). Returns a GeoJSON FeatureCollection of
    Points carrying an `intensity_mm_hr` property, ready for a Mapbox
    heatmap layer."""
    lats = [
        bounds["south"] + (bounds["north"] - bounds["south"]) * i / (resolution - 1)
        for i in range(resolution)
    ]
    lons = [
        bounds["west"] + (bounds["east"] - bounds["west"]) * i / (resolution - 1)
        for i in range(resolution)
    ]

    features = []
    for lat in lats:
        for lon in lons:
            intensity = intensity_mm_per_hr(lon, lat, elapsed_min, storm)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"intensity_mm_hr": round(intensity, 2)},
                }
            )

    return {
        "type": "FeatureCollection",
        "features": features,
        "elapsed_min": elapsed_min,
        "storm_center": list(storm_center_at(elapsed_min, storm)),
    }


if __name__ == "__main__":
    cx, cy = AOI_CENTER
    for t in (0, 45, 90, 135, 180):
        at_center = intensity_mm_per_hr(cx, cy, t)
        far_away = intensity_mm_per_hr(cx + 0.05, cy, t)
        print(f"t={t:>3}min  at-center={at_center:6.2f}mm/hr  2.5km-away={far_away:6.2f}mm/hr")
