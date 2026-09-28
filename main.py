"""FastAPI backend for the urban flood nowcasting API."""

import logging
from dataclasses import replace
from datetime import timedelta

from fastapi import FastAPI, Response, UploadFile, File, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
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
from dimension_ingest import ingest_survey

import os
import tempfile


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
    """Layer 1 (authoritative): run PySWMM and return a 0-3hr nowcast time series of street-level flood depths.

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

        response.headers["X-Engine"] = "physics"
        return {"frames": frames}

    except Exception as exc:
        # This is the authoritative output shown to decision-makers, so a
        # crashed engine must surface as an error, never as stand-in depths.
        logger.exception("PySWMM engine failed")
        raise HTTPException(
            status_code=503,
            detail=f"Physics engine (PySWMM) failed: {exc}. No flood depths are available for this run.",
        ) from exc


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


# ---------------------------------------------------------------------------
# Admin: pipe survey upload
# ---------------------------------------------------------------------------
# Protect the admin endpoint with a simple API key (set ADMIN_API_KEY in .env).
# Government operators hit this endpoint; the system does the rest.
# ---------------------------------------------------------------------------
_ADMIN_KEY_HEADER = APIKeyHeader(name="X-Admin-Key", auto_error=False)
_ADMIN_KEY = os.getenv("ADMIN_API_KEY", "")


def _require_admin(key: str = Security(_ADMIN_KEY_HEADER)) -> None:
    if not _ADMIN_KEY:
        raise HTTPException(
            status_code=503,
            detail="Admin API key not configured on this server (set ADMIN_API_KEY in .env).",
        )
    if key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")


class DimensionUpdateResponse(BaseModel):
    rows_parsed: int
    changes_applied: int
    warnings: List[str]
    errors: List[str]
    backup_path: str | None
    dry_run: bool
    message: str


@app.post(
    "/admin/update-dimensions",
    response_model=DimensionUpdateResponse,
    summary="Upload a pipe survey CSV to update drainage dimensions",
    description=(
        "Accepts a CSV with columns: pipe_id, diameter_mm, shape (optional), "
        "material (optional), condition (optional), survey_date (optional), surveyor (optional). "
        "Validates the data, shows what will change, writes a versioned backup of pune_base.inp, "
        "patches pipe diameters and Manning's roughness, and triggers recalibration. "
        "Requires X-Admin-Key header."
    ),
)
async def update_dimensions(
    file: UploadFile = File(..., description="Pipe survey CSV file"),
    dry_run: bool = False,
    _: None = Depends(_require_admin),
) -> DimensionUpdateResponse:
    """Government-facing endpoint: upload a surveyed pipe dimension CSV.

    The system diffs the CSV against the current network, applies changes,
    keeps a timestamped backup, and writes an audit log — no SWMM or Python
    knowledge required on the operator's side.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Uploaded file must be a .csv")

    # Write the upload to a temp file so dimension_ingest can read it normally.
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest_survey(tmp_path, dry_run=dry_run, run_recalibration=not dry_run)
    finally:
        os.unlink(tmp_path)

    changes_applied = len(result["changes"]) if not dry_run else 0
    message = (
        f"Dry run: {len(result['changes'])} change(s) would be applied."
        if dry_run
        else (
            f"{changes_applied} change(s) applied. Backup: {result['backup_path']}"
            if changes_applied
            else "No changes — network already matches the survey data."
        )
    )

    return DimensionUpdateResponse(
        rows_parsed=result["rows_parsed"],
        changes_applied=changes_applied,
        warnings=result["warnings"],
        errors=result["errors"],
        backup_path=result["backup_path"],
        dry_run=dry_run,
        message=message,
    )


# ---------------------------------------------------------------------------
# GNN inference endpoint
# ---------------------------------------------------------------------------
# Drop-in replacement for /simulate using the trained FloodGNN.
# Returns the same NowcastResponse format so the frontend works unchanged.
# The predictor is loaded lazily on first request — if no checkpoint exists,
# the endpoint returns a clear error instead of crashing the server.
# ---------------------------------------------------------------------------

_gnn_predictor = None
_GNN_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "flood_gnn_best.pt")


def _get_gnn_predictor():
    """Lazily load the GNN predictor on first call."""
    global _gnn_predictor
    if _gnn_predictor is None:
        if not os.path.exists(_GNN_CHECKPOINT):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"GNN checkpoint not found at {_GNN_CHECKPOINT}. "
                    "Run: python3 generate_training_data.py && python3 -m gnn.train"
                ),
            )
        from gnn.inference import FloodGNNPredictor
        _gnn_predictor = FloodGNNPredictor(_GNN_CHECKPOINT)
    return _gnn_predictor


@app.post("/simulate_gnn", response_model=NowcastResponse)
def simulate_gnn(request: SimulationRequest, response: Response) -> NowcastResponse:
    """GNN-powered flood nowcast: ~50ms inference vs ~2s for SWMM.

    Uses the trained FloodGNN to predict per-node surcharge volumes,
    then pipes them through the same bathtub-fill post-processing
    (flood_fill.py) as /simulate for per-road flood depths.

    Returns the same NowcastResponse format as /simulate — the frontend
    can switch between them without any code changes.
    """
    import time as _time

    t0 = _time.perf_counter()

    try:
        predictor = _get_gnn_predictor()
        gnn_frames = predictor.predict(rainfall_mm_hr=request.rainfall_mm_per_hr)
    except HTTPException:
        raise
    except Exception:
        logger.exception("GNN inference failed")
        raise HTTPException(status_code=500, detail="GNN inference failed")

    t_gnn = _time.perf_counter()

    # Post-process: map predicted surcharge volumes → road-level flood depths
    # using the same bathtub-fill as /simulate
    frames: list[dict] = []
    for gnn_frame in gnn_frames:
        surcharging_nodes = [
            {"node_id": nid, "surcharge_volume_m3": vol}
            for nid, vol in gnn_frame["node_surcharges"].items()
        ]
        depths = calculate_area_depths(surcharging_nodes)
        frames.append({
            "elapsed_min": gnn_frame["elapsed_min"],
            "depths": depths,
        })

    t_total = _time.perf_counter()
    response.headers["X-Engine"] = "gnn"
    response.headers["X-GNN-Inference-Ms"] = f"{(t_gnn - t0) * 1000:.1f}"
    response.headers["X-Total-Ms"] = f"{(t_total - t0) * 1000:.1f}"

    return {"frames": frames}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
