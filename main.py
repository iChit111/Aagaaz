"""FastAPI backend for the urban flood nowcasting API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

from urban_flood_schema import (
    FloodFeatureCollection,
    make_node,
    make_pipe,
)


class SimulationRequest(BaseModel):
    rainfall_mm_per_hr: float


# Offline replacements for the precomputed spatial joins.
NODE_TO_ROAD = {
    "MANHOLE-PUNE-001": "MG_ROAD_SEG_3",
    "MANHOLE-PUNE-002": "FC_ROAD_SEG_1",
}
ROAD_AREA_SQM = {
    "MG_ROAD_SEG_3": 1200.0,
    "FC_ROAD_SEG_1": 1000.0,
}
RoadFloodStatus = Dict[str, float]


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


def calculate_road_depths(surcharging_nodes: list[dict]) -> dict:
    """Map node surcharge volumes to road flood depths in centimeters."""
    road_depths = {}
    for node in surcharging_nodes:
        road_id = NODE_TO_ROAD.get(node["node_id"])
        if road_id is None:
            continue

        road_area_sqm = ROAD_AREA_SQM[road_id]
        depth_cm = node["surcharge_volume_m3"] / road_area_sqm * 100
        road_depths[road_id] = depth_cm

    return road_depths


@app.post("/simulate", response_model=RoadFloodStatus)
def simulate(request: SimulationRequest) -> RoadFloodStatus:
    """Return road flood depths derived from mock surcharge volumes."""
    surcharge_volume_m3 = 184.8 if request.rainfall_mm_per_hr > 50 else 0.0
    surcharging_nodes = [
        {
            "node_id": "MANHOLE-PUNE-001",
            "surcharge_volume_m3": surcharge_volume_m3,
        },
        {
            "node_id": "MANHOLE-PUNE-002",
            "surcharge_volume_m3": 0.0,
        },
    ]
    return calculate_road_depths(surcharging_nodes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
