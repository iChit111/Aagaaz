"""Flood-aware shortest-path routing over the Pune road network.

Builds an undirected graph from the same pune_roads.json GeoJSON the
frontend renders, then routes around roads the current nowcast frame
marks as flooded — the API utility that lets navigation, emergency
services, and commuters steer around street-level inundation.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROADS_PATH = os.path.join(BASE_DIR, "frontend", "src", "pune_roads.json")

# Roads at/above this depth are excluded from routing wherever a detour exists.
IMPASSABLE_CM = 30.0
# Roads at/above this depth are passable but discouraged (pooling).
WARNING_CM = 10.0
WARNING_PENALTY_MULTIPLIER = 6.0
# Multiplier applied to impassable segments only when no dry path exists at all.
STRANDED_PENALTY_MULTIPLIER = 50.0

Coordinate = Tuple[float, float]  # (lon, lat), rounded to dedupe shared endpoints


class NoRouteFound(Exception):
    """Raised when origin and destination are not connected in the road graph."""


def _round(coord: List[float]) -> Coordinate:
    return (round(coord[0], 7), round(coord[1], 7))


def _haversine_m(a: Coordinate, b: Coordinate) -> float:
    """Great-circle distance in meters between two (lon, lat) points."""
    lon1, lat1, lon2, lat2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def build_road_graph(filepath: str = ROADS_PATH) -> nx.Graph:
    """Build a routable graph: nodes are junction coordinates, edges are
    road segments tagged with the id of the road (way) they belong to."""
    with open(filepath, "r") as f:
        data = json.load(f)

    graph = nx.Graph()
    for feature in data.get("features", []):
        road_id = feature.get("id") or feature.get("properties", {}).get("id")
        coords = feature.get("geometry", {}).get("coordinates", [])
        if not road_id or len(coords) < 2:
            continue
        for start, end in zip(coords, coords[1:]):
            u, v = _round(start), _round(end)
            if u == v:
                continue
            length_m = _haversine_m(u, v)
            if graph.has_edge(u, v) and graph[u][v]["length_m"] <= length_m:
                continue
            graph.add_edge(u, v, length_m=length_m, road_id=road_id)

    return graph


def _nearest_node(graph: nx.Graph, point: Coordinate) -> Coordinate:
    return min(graph.nodes, key=lambda node: _haversine_m(node, point))


def _edge_weight(length_m: float, road_id: str, depths: Dict[str, float]) -> Optional[float]:
    """Weight a segment by flood depth. None means impassable."""
    depth_cm = depths.get(road_id, 0.0)
    if depth_cm >= IMPASSABLE_CM:
        return None
    if depth_cm >= WARNING_CM:
        return length_m * WARNING_PENALTY_MULTIPLIER
    return length_m


def _make_weight_fn(depths: Dict[str, float], allow_impassable: bool):
    def weight(u, v, edge_data):
        w = _edge_weight(edge_data["length_m"], edge_data["road_id"], depths)
        if w is not None:
            return w
        return edge_data["length_m"] * STRANDED_PENALTY_MULTIPLIER if allow_impassable else None

    return weight


def find_route(
    graph: nx.Graph,
    origin: Coordinate,
    destination: Coordinate,
    depths: Dict[str, float],
) -> dict:
    """Return the flood-aware shortest path between origin and destination.

    Tries a route that avoids impassable (>=30cm) roads entirely first; if
    flooding has cut the area off, retries allowing impassable segments
    (heavily penalized) and flags the result as degraded so callers can
    warn the driver instead of silently routing through floodwater.
    """
    start = _nearest_node(graph, origin)
    end = _nearest_node(graph, destination)

    try:
        path = nx.shortest_path(graph, start, end, weight=_make_weight_fn(depths, allow_impassable=False))
        degraded = False
    except nx.NetworkXNoPath:
        try:
            path = nx.shortest_path(graph, start, end, weight=_make_weight_fn(depths, allow_impassable=True))
            degraded = True
        except nx.NetworkXNoPath as exc:
            raise NoRouteFound("No path exists between origin and destination") from exc

    road_ids: List[str] = []
    flooded_road_ids: List[str] = []
    distance_m = 0.0
    for u, v in zip(path, path[1:]):
        edge = graph[u][v]
        distance_m += edge["length_m"]
        road_ids.append(edge["road_id"])
        if depths.get(edge["road_id"], 0.0) >= WARNING_CM:
            flooded_road_ids.append(edge["road_id"])

    return {
        "path": [[lon, lat] for lon, lat in path],
        "distance_m": round(distance_m, 1),
        "road_ids": road_ids,
        "flooded_road_ids": sorted(set(flooded_road_ids)),
        "degraded": degraded,
    }
