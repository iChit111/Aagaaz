"""Typed schema and GeoJSON serialization for urban flood nowcasting."""

from __future__ import annotations

import json
from typing import Literal
from typing_extensions import TypedDict

Coordinate = tuple[float, float]


class PointGeometry(TypedDict):
    type: Literal["Point"]
    coordinates: Coordinate


class LineStringGeometry(TypedDict):
    type: Literal["LineString"]
    coordinates: list[Coordinate]


class NodeProperties(TypedDict):
    node_id: str
    elevation: float
    surcharge_depth_cm: float
    status: Literal["SAFE", "WARNING", "FLOODED"]


class PipeProperties(TypedDict):
    pipe_id: str
    flow_rate_lps: float
    max_capacity_lps: float
    utilization_pct: float
    status: Literal["NORMAL", "SURCHARGING"]


class NodeFeature(TypedDict):
    type: Literal["Feature"]
    geometry: PointGeometry
    properties: NodeProperties


class PipeFeature(TypedDict):
    type: Literal["Feature"]
    geometry: LineStringGeometry
    properties: PipeProperties


class FloodFeatureCollection(TypedDict):
    type: Literal["FeatureCollection"]
    features: list[NodeFeature | PipeFeature]


def make_node(
    node_id: str,
    elevation: float,
    surcharge_depth_cm: float,
    coordinates: Coordinate,
) -> NodeFeature:
    """Build a drainage node and derive its status from surcharge depth."""
    if surcharge_depth_cm < 0:
        raise ValueError("surcharge_depth_cm cannot be negative")

    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": coordinates},
        "properties": {
            "node_id": node_id,
            "elevation": elevation,
            "surcharge_depth_cm": surcharge_depth_cm,
            "status": "FLOODED" if surcharge_depth_cm > 0 else "SAFE",
        },
    }


def make_pipe(
    pipe_id: str,
    flow_rate_lps: float,
    max_capacity_lps: float,
    coordinates: list[Coordinate],
) -> PipeFeature:
    """Build an underground pipe and derive utilization and status."""
    if flow_rate_lps < 0:
        raise ValueError("flow_rate_lps cannot be negative")
    if max_capacity_lps <= 0:
        raise ValueError("max_capacity_lps must be greater than zero")
    if len(coordinates) < 2:
        raise ValueError("a LineString needs at least two coordinates")

    utilization_pct = flow_rate_lps / max_capacity_lps * 100
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "properties": {
            "pipe_id": pipe_id,
            "flow_rate_lps": flow_rate_lps,
            "max_capacity_lps": max_capacity_lps,
            "utilization_pct": utilization_pct,
            "status": "SURCHARGING" if utilization_pct >= 100 else "NORMAL",
        },
    }


def to_geojson(feature_collection: FloodFeatureCollection, *, indent: int | None = None) -> str:
    """Serialize a FeatureCollection as a GeoJSON string for Mapbox GL JS."""
    return json.dumps(
        feature_collection,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def create_dummy_payload() -> FloodFeatureCollection:
    """Create one flooded node and one surcharging pipe for API examples."""
    node = make_node(
        node_id="MANHOLE-001",
        elevation=12.4,
        surcharge_depth_cm=8.5,
        coordinates=(73.8567, 18.5204),
    )
    pipe = make_pipe(
        pipe_id="PIPE-001",
        flow_rate_lps=1250.0,
        max_capacity_lps=1000.0,
        coordinates=[(73.8567, 18.5204), (73.8581, 18.5212)],
    )
    return {"type": "FeatureCollection", "features": [node, pipe]}


def main() -> None:
    print(to_geojson(create_dummy_payload(), indent=2))


if __name__ == "__main__":
    main()
