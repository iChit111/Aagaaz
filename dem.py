"""Fetches and caches the SRTM 30m DEM tile covering the Pune AOI.

The AOI matches the extent of frontend/src/pune_roads.json, padded ~0.01
degrees (~1km) so catchments delineated near the map edges (Phase B)
aren't artificially cut off by the DEM boundary.
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_PATH = os.path.join(DATA_DIR, "dem_pune.tif")

OPENTOPOGRAPHY_URL = "https://portal.opentopography.org/API/globaldem"
AOI_BOUNDS = {"south": 18.497, "north": 18.535, "west": 73.819, "east": 73.864}


class DemFetchError(Exception):
    pass


def fetch_dem(dest_path: str = DEM_PATH, *, force: bool = False) -> str:
    """Download the SRTM GL1 (30m) DEM for the AOI, caching it locally."""
    if os.path.exists(dest_path) and not force:
        return dest_path

    api_key = os.getenv("OPENTOPOGRAPHY_API_KEY")
    if not api_key:
        raise DemFetchError(
            "OPENTOPOGRAPHY_API_KEY is not set. Add it to .env (see .env.example)."
        )

    params = {
        "demtype": "SRTMGL1",
        "outputFormat": "GTiff",
        "API_Key": api_key,
        **AOI_BOUNDS,
    }
    response = requests.get(OPENTOPOGRAPHY_URL, params=params, timeout=60)
    if response.status_code != 200 or not response.content:
        raise DemFetchError(
            f"OpenTopography request failed ({response.status_code}): {response.text[:300]}"
        )
    # A failed request sometimes still returns 200 with a small JSON/HTML error body
    # instead of GeoTIFF bytes -- guard against silently caching a bad file.
    if response.content[:4] not in (b"II*\x00", b"MM\x00*"):
        raise DemFetchError(f"Response was not a GeoTIFF: {response.content[:300]!r}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(response.content)
    return dest_path


if __name__ == "__main__":
    path = fetch_dem(force=True)
    print(f"DEM saved to {path}")

    import rasterio

    with rasterio.open(path) as src:
        print(f"shape={src.shape} crs={src.crs} bounds={src.bounds}")
        band = src.read(1)
        print(f"elevation min={band.min():.1f}m max={band.max():.1f}m")
