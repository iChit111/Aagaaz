"""FastAPI backend for the urban flood nowcasting API."""

import logging
from dataclasses import replace
from datetime import timedelta

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple
from fastapi import HTTPException
from pyswmm import Simulation, errors

logger = logging.getLogger(__name__)

from urban_flood_schema import (
    FloodFeatureCollection,
    make_node,
    make_pipe,
)
from routing import NoRouteFound, build_road_graph, find_route
from network_topology import JUNCTION_COORDINATES
from nowcast import DEFAULT_STORM, sample_grid
from dem import AOI_BOUNDS
from runoff import CatchmentRunoff
from flood_fill import FloodExtentEstimator

import os


class SimulationRequest(BaseModel):
    rainfall_mm_per_hr: float


# Anchor to this file's location, not the process's cwd.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROAD_GRAPH = build_road_graph()
RUNOFF = CatchmentRunoff()
FLOOD_EXTENT = FloodExtentEstimator()

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


def _parse_inp_network(inp_path: str) -> tuple[dict[str, float], list[tuple[str, str]]]:
    """Read [JUNCTIONS] elevations and [CONDUITS] node pairs straight out
    of the .inp file, so the dry-network view reflects whatever network
    generate_network.py last produced without a second source of truth."""
    elevations: dict[str, float] = {}
    conduits: list[tuple[str, str]] = []
    section = None
    with open(inp_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("["):
                section = line.strip("[]")
                continue
            parts = line.split()
            if section == "JUNCTIONS":
                elevations[parts[0]] = float(parts[1])
            elif section == "CONDUITS":
                conduits.append((parts[1], parts[2]))
    return elevations, conduits


def build_dry_network() -> FloodFeatureCollection:
    """Build the baseline dry drainage network for Pune."""
    elevations, conduits = _parse_inp_network(os.path.join(BASE_DIR, "pune_base.inp"))

    features = [
        make_node(
            node_id=node_id,
            elevation=elevations.get(node_id, 0.0),
            surcharge_depth_cm=0.0,
            coordinates=coordinates,
        )
        for node_id, coordinates in JUNCTION_COORDINATES.items()
    ]
    for pipe_id, (node1, node2) in enumerate(conduits):
        if node1 not in JUNCTION_COORDINATES or node2 not in JUNCTION_COORDINATES:
            continue  # skip stub pipes to synthetic OUTFALL nodes, which have no lon/lat
        features.append(
            make_pipe(
                pipe_id=f"PIPE-{pipe_id}",
                flow_rate_lps=0.0,
                max_capacity_lps=1000.0,
                coordinates=[JUNCTION_COORDINATES[node1], JUNCTION_COORDINATES[node2]],
            )
        )
    return {"type": "FeatureCollection", "features": features}


@app.get("/network", response_model=FloodFeatureCollection)
def get_network() -> FloodFeatureCollection:
    """Return the current dry baseline network."""
    return build_dry_network()


def calculate_area_depths(surcharging_nodes: list[dict]) -> dict:
    """Map each node's surcharge volume to real street-level depths via a
    DEM bathtub-fill (flood_fill.py) rather than spreading it evenly
    across a fixed, hand-picked road cluster. Where two nodes' pools
    reach the same road, the deeper estimate wins."""
    cell_depth_cm: dict[tuple[int, int], float] = {}
    for node in surcharging_nodes:
        extent = FLOOD_EXTENT.flood_extent_by_index(node["node_id"], node["surcharge_volume_m3"])
        for index, depth_cm in extent.items():
            if depth_cm > cell_depth_cm.get(index, 0.0):
                cell_depth_cm[index] = depth_cm

    if not cell_depth_cm:
        return {}
    return FLOOD_EXTENT.road_depths(cell_depth_cm)


def _snapshot(elapsed_min: int, nodes: dict) -> FloodFrame:
    surcharging_nodes = [
        {"node_id": node_id, "surcharge_volume_m3": node.statistics["flooding_volume"]}
        for node_id, node in nodes.items()
    ]
    return {"elapsed_min": elapsed_min, "depths": calculate_area_depths(surcharging_nodes)}


@app.post("/simulate", response_model=NowcastResponse)
def simulate(request: SimulationRequest, response: Response) -> NowcastResponse:
    """Run PySWMM and return a 0-3hr nowcast time series of street-level flood depths.

    Rainfall is driven by nowcast.py's synthetic storm cell, scaled so its
    peak matches the requested rainfall_mm_per_hr. Each junction gets its
    own time-varying inflow computed from its DEM-derived catchment
    (terrain.py / runoff.py) rather than one manual scalar hardcoded onto
    a single node.
    """
    try:
        from pyswmm import Simulation, Nodes

        storm = replace(DEFAULT_STORM, peak_intensity_mm_hr=request.rainfall_mm_per_hr)

        with Simulation(os.path.join(BASE_DIR, "pune_base.inp")) as sim:
            nodes = {node_id: Nodes(sim)[node_id] for node_id in JUNCTION_COORDINATES}

            start_time = sim.start_time
            next_report = start_time
            frames = []

            # Step through the physics, refreshing each node's catchment-derived
            # inflow every routing step and capturing a frame every report interval.
            # node.statistics can only be read once the engine has started stepping,
            # so the t=0 baseline frame is taken on the loop's first iteration.
            for _step in sim:
                elapsed_min = (sim.current_time - start_time).total_seconds() / 60
                for node_id, node in nodes.items():
                    node.generated_inflow(RUNOFF.inflow_cms(node_id, elapsed_min, storm))

                if not frames:
                    frames.append(_snapshot(0, nodes))
                if sim.current_time >= next_report + REPORT_INTERVAL:
                    next_report += REPORT_INTERVAL
                    report_elapsed = int((next_report - start_time).total_seconds() // 60)
                    frames.append(_snapshot(report_elapsed, nodes))

            final_elapsed = int((sim.current_time - start_time).total_seconds() // 60)
            if final_elapsed != frames[-1]["elapsed_min"]:
                frames.append(_snapshot(final_elapsed, nodes))

        return {"frames": frames}

    except Exception:
        logger.exception("PySWMM engine failed; falling back to mock flood data")
        response.headers["X-Engine-Status"] = "fallback"
        # Bulletproof fallback: flag every junction as surcharging once rainfall
        # crosses a rough heavy-rain threshold, so a crashed engine still shows
        # *something* plausible instead of an empty map.
        surcharge_volume_m3 = 5000.0 if request.rainfall_mm_per_hr > 50 else 0.0
        surcharging_nodes = [
            {"node_id": node_id, "surcharge_volume_m3": surcharge_volume_m3}
            for node_id in JUNCTION_COORDINATES
        ]
        return {"frames": [{"elapsed_min": 0, "depths": calculate_area_depths(surcharging_nodes)}]}


class RouteRequest(BaseModel):
    origin: Tuple[float, float]  # [lon, lat]
    destination: Tuple[float, float]
    depths: RoadFloodStatus = Field(default_factory=dict)


class RouteResponse(BaseModel):
    path: List[List[float]]
    distance_m: float
    road_ids: List[str]
    flooded_road_ids: List[str]
    degraded: bool


@app.get("/nowcast")
def nowcast(elapsed_min: float = 0, resolution: int = 15) -> dict:
    """Synthetic rainfall nowcast: a spatial intensity grid at a point in
    the 0-3hr window. Stands in for a live Doppler radar nowcast -- see
    nowcast.py for why, and how to swap in a real feed later."""
    if resolution < 2:
        raise HTTPException(status_code=422, detail="resolution must be >= 2")
    return sample_grid(AOI_BOUNDS, elapsed_min, resolution)


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> RouteResponse:
    """Find a flood-safe path between two points, avoiding roads the
    current nowcast frame marks as flooded wherever a detour exists."""
    try:
        return find_route(ROAD_GRAPH, request.origin, request.destination, request.depths)
    except NoRouteFound as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
