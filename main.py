"""FastAPI backend for the urban flood nowcasting API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from urban_flood_schema import (
    FloodFeatureCollection,
    create_dummy_payload,
    make_node,
    make_pipe,
)


class SimulationRequest(BaseModel):
    rainfall_mm_per_hr: float


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


@app.post("/simulate", response_model=FloodFeatureCollection)
def simulate(request: SimulationRequest) -> FloodFeatureCollection:
    """Return a placeholder flooded state for heavy rainfall."""
    if request.rainfall_mm_per_hr > 50:
        return create_dummy_payload()
    return build_dry_network()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
