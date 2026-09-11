"""FastAPI backend for the urban flood nowcasting API."""

import logging
from datetime import timedelta

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from fastapi import HTTPException
from pyswmm import Simulation, errors

logger = logging.getLogger(__name__)

from urban_flood_schema import (
    FloodFeatureCollection,
    make_node,
    make_pipe,
)


class SimulationRequest(BaseModel):
    rainfall_mm_per_hr: float


# Offline replacements for the precomputed spatial joins.
import json
import os

def load_road_cluster(filepath: str, cluster_size: int = 40):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        road_ids = []
        for feature in data.get('features', [])[:cluster_size]:
            road_id = feature.get('id') or feature.get('properties', {}).get('id')
            if road_id:
                road_ids.append(road_id)
        if not road_ids:
            print(f"WARNING: {filepath} loaded but produced zero road IDs.")
        return road_ids
    except Exception as e:
        # Fail loudly at startup — this should never be silently swallowed,
        # because an empty flood zone makes /simulate a permanent no-op.
        raise RuntimeError(f"Could not load road cluster from {filepath}: {e}") from e

# Anchor to this file's location, not the process's cwd.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEOJSON_PATH = os.path.join(BASE_DIR, "frontend", "src", "pune_roads.json")
DECCAN_FLOOD_ZONE = load_road_cluster(GEOJSON_PATH, cluster_size=15)

assert DECCAN_FLOOD_ZONE, "DECCAN_FLOOD_ZONE is empty — /simulate will always return {}"

NODE_TO_AREA = {
    "MANHOLE-PUNE-001": DECCAN_FLOOD_ZONE
}

# Assume average road segment area is 500 sqm
AVERAGE_ROAD_AREA_SQM = 500.0
RoadFloodStatus = Dict[str, float]
REPORT_INTERVAL = timedelta(minutes=15)


class FloodFrame(BaseModel):
    elapsed_min: int
    depths: RoadFloodStatus


class NowcastResponse(BaseModel):
    frames: List[FloodFrame]

app = FastAPI(title="Urban Flood Nowcasting API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_dry_network() -> FloodFeatureCollection:
    """Build the baseline dry drainage network for Pune."""
    node = make_node(
        node_id="MANHOLE-PUNE-001",
        elevation=560.0,
        surcharge_depth_cm=0.0,
        coordinates=(73.8567, 18.5204),
    )
    pipe = make_pipe(
        pipe_id="PIPE-PUNE-001",
        flow_rate_lps=0.0,
        max_capacity_lps=1000.0,
        coordinates=[(73.8567, 18.5204), (73.8581, 18.5212)],
    )
    return {"type": "FeatureCollection", "features": [node, pipe]}


@app.get("/network", response_model=FloodFeatureCollection)
def get_network() -> FloodFeatureCollection:
    """Return the current dry baseline network."""
    return build_dry_network()


def calculate_area_depths(surcharging_nodes: list[dict]) -> dict:
    """Distribute surcharge volume across an entire area of roads."""
    road_depths = {}
    for node in surcharging_nodes:
        impacted_roads = NODE_TO_AREA.get(node["node_id"], [])
        if not impacted_roads:
            continue

        # Total volume divided by the total area of ALL flooded roads
        total_area_sqm = len(impacted_roads) * AVERAGE_ROAD_AREA_SQM
        depth_cm = (node["surcharge_volume_m3"] / total_area_sqm) * 100
        
        # Assign this depth to every road in the cluster
        for road_id in impacted_roads:
            road_depths[road_id] = depth_cm

    return road_depths


def _snapshot(elapsed_min: int, manhole_1, manhole_2) -> FloodFrame:
    vol_1 = manhole_1.statistics["flooding_volume"]
    vol_2 = manhole_2.statistics["flooding_volume"]
    surcharging_nodes = [
        {"node_id": "MANHOLE-PUNE-001", "surcharge_volume_m3": vol_1},
        {"node_id": "MANHOLE-PUNE-002", "surcharge_volume_m3": vol_2},
    ]
    return {"elapsed_min": elapsed_min, "depths": calculate_area_depths(surcharging_nodes)}


@app.post("/simulate", response_model=NowcastResponse)
def simulate(request: SimulationRequest, response: Response) -> NowcastResponse:
    """Run PySWMM and return a 0-3hr nowcast time series of street-level flood depths."""
    try:
        from pyswmm import Simulation, Nodes

        with Simulation(os.path.join(BASE_DIR, "pune_base.inp")) as sim:
            manhole_1 = Nodes(sim)["MANHOLE-PUNE-001"]
            manhole_2 = Nodes(sim)["MANHOLE-PUNE-002"]

            # Convert UI slider (mm/hr) to Inflow (CMS) for a 1-hectare catchment
            runoff_cms = (request.rainfall_mm_per_hr / 3600000) * 10000 * 0.9
            manhole_1.generated_inflow(runoff_cms)

            start_time = sim.start_time
            next_report = start_time
            frames = []

            # Step through the physics, capturing a frame every report interval.
            # node.statistics can only be read once the engine has started stepping,
            # so the t=0 baseline frame is taken on the loop's first iteration.
            for _step in sim:
                if not frames:
                    frames.append(_snapshot(0, manhole_1, manhole_2))
                if sim.current_time >= next_report + REPORT_INTERVAL:
                    next_report += REPORT_INTERVAL
                    elapsed_min = int((next_report - start_time).total_seconds() // 60)
                    frames.append(_snapshot(elapsed_min, manhole_1, manhole_2))

            final_elapsed = int((sim.current_time - start_time).total_seconds() // 60)
            if final_elapsed != frames[-1]["elapsed_min"]:
                frames.append(_snapshot(final_elapsed, manhole_1, manhole_2))

        return {"frames": frames}

    except Exception:
        logger.exception("PySWMM engine failed; falling back to mock flood data")
        response.headers["X-Engine-Status"] = "fallback"
        # Bulletproof Fallback: Use the mock logic you already wrote!
        surcharge_volume_m3 = 5000.0 if request.rainfall_mm_per_hr > 50 else 0.0
        surcharging_nodes = [
            {"node_id": "MANHOLE-PUNE-001", "surcharge_volume_m3": surcharge_volume_m3},
            {"node_id": "MANHOLE-PUNE-002", "surcharge_volume_m3": 0.0},
        ]
        return {"frames": [{"elapsed_min": 0, "depths": calculate_area_depths(surcharging_nodes)}]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
