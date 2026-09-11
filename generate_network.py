"""Generates a larger, realistic SWMM drainage network for the demo.

The original pune_base.inp had 2 hand-placed junctions -- enough to prove
the pipeline works, but too sparse to visually show "many streets
flooding" for a hackathon demo. This picks real intersections from the
road graph (routing.py), looks up their real DEM elevation, and connects
each to its nearest downhill neighbor (a greedy minimum-spanning-tree-like
topology) to build a plausible branching network converging to an outfall
-- instead of hand-authoring more .inp entries one at a time.

Run standalone to regenerate pune_base.inp and network_topology.py, then
re-run terrain.py to delineate catchments for the new junctions.
"""

from __future__ import annotations

import random

import networkx as nx
import rasterio

from dem import DEM_PATH
from routing import _haversine_m, build_road_graph

N_JUNCTIONS = 25
MIN_JUNCTION_DEGREE = 3  # only place junctions at real intersections, not mid-block
BASE_PIPE_DIAMETER_M = 0.25
DIAMETER_PER_UPSTREAM_M = 0.03
MAX_PIPE_DIAMETER_M = 1.2
OUTFALL_DROP_M = 3.0  # elevation drop from the lowest junction to its outfall


def select_junctions(n: int = N_JUNCTIONS) -> list[tuple[float, float]]:
    """Farthest-point sample real intersections for good spatial spread."""
    g = build_road_graph()
    largest = max(nx.connected_components(g), key=len)
    sub = g.subgraph(largest)
    degrees = dict(sub.degree())
    candidates = [node for node, d in degrees.items() if d >= MIN_JUNCTION_DEGREE]

    random.seed(42)  # reproducible layout across regenerations
    selected = [candidates[0]]
    remaining = set(candidates[1:])
    while len(selected) < n and remaining:
        best = max(remaining, key=lambda p: min(_haversine_m(p, s) for s in selected))
        selected.append(best)
        remaining.discard(best)
    return selected


def build_topology(junctions: dict[str, dict]) -> tuple[list[tuple[str, str, float]], list[str]]:
    """Connect each junction to its nearest strictly-lower-elevation
    neighbor. Junctions with no lower neighbor are local minima -- outfalls.
    Returns (edges as (upstream, downstream, length_m), outfall node ids).
    """
    names = list(junctions)
    edges = []
    outfalls = []
    for name in names:
        j = junctions[name]
        lower = [
            (other, _haversine_m((j["lon"], j["lat"]), (junctions[other]["lon"], junctions[other]["lat"])))
            for other in names
            if other != name and junctions[other]["elev"] < j["elev"]
        ]
        if not lower:
            outfalls.append(name)
            continue
        nearest_id, length_m = min(lower, key=lambda t: t[1])
        edges.append((name, nearest_id, length_m))
    return edges, outfalls


def _upstream_counts(edges: list[tuple[str, str, float]], all_nodes: list[str]) -> dict[str, int]:
    """Number of junctions draining through each node (including itself) --
    used to size pipes larger as they carry more accumulated flow."""
    children: dict[str, list[str]] = {n: [] for n in all_nodes}
    for u, v, _ in edges:
        children[v].append(u)

    counts: dict[str, int] = {}

    def count(node: str) -> int:
        if node in counts:
            return counts[node]
        total = 1 + sum(count(child) for child in children[node])
        counts[node] = total
        return total

    for n in all_nodes:
        count(n)
    return counts


def generate() -> tuple[dict[str, dict], list[tuple[str, str, float]], dict[str, str]]:
    """Returns (junctions, edges, outfall_pipe_targets).

    outfall_pipe_targets maps each local-minimum junction name to a
    synthetic OUTFALL node name it drains into.
    """
    with rasterio.open(DEM_PATH) as src:
        elevation = src.read(1).astype(float)
        transform = src.transform

    junctions: dict[str, dict] = {}
    for i, (lon, lat) in enumerate(select_junctions()):
        row, col = rasterio.transform.rowcol(transform, lon, lat)
        junctions[f"J{i:02d}"] = {"lon": lon, "lat": lat, "elev": float(elevation[row, col])}

    edges, outfall_junctions = build_topology(junctions)
    outfall_targets = {name: f"OUTFALL-{name}" for name in outfall_junctions}
    return junctions, edges, outfall_targets


def render_inp(junctions, edges, outfall_targets) -> str:
    counts = _upstream_counts(edges, list(junctions))

    lines = [
        "[TITLE]",
        "Pune Deccan Gymkhana Network",
        "",
        "[OPTIONS]",
        "FLOW_UNITS CMS",
        "FLOW_ROUTING KINWAVE",
        "START_DATE 01/01/2026",
        "START_TIME 00:00:00",
        "END_DATE 01/01/2026",
        "END_TIME 03:00:00",
        "",
        "[JUNCTIONS]",
        ";;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded",
    ]
    for name, j in junctions.items():
        lines.append(f"{name:<16} {j['elev']:<10.1f} 1.5        0          0.3        0")

    lines += ["", "[OUTFALLS]", ";;Name           Elevation  Type       Stage Data       Gated"]
    for junction_name, outfall_name in outfall_targets.items():
        outfall_elev = junctions[junction_name]["elev"] - OUTFALL_DROP_M
        lines.append(f"{outfall_name:<16} {outfall_elev:<10.1f} FREE                        NO")

    lines += [
        "",
        "[CONDUITS]",
        ";;Name           Node1            Node2            Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow",
    ]
    for u, v, length_m in edges:
        pipe_name = f"PIPE-{u}-{v}"
        lines.append(f"{pipe_name:<16} {u:<16} {v:<16} {length_m:<10.1f} 0.013      0          0          0          0")
    # Short stub pipe from each local-minimum junction down to its synthetic outfall.
    for junction_name, outfall_name in outfall_targets.items():
        pipe_name = f"PIPE-{junction_name}-OUT"
        lines.append(f"{pipe_name:<16} {junction_name:<16} {outfall_name:<16} 50.0       0.013      0          0          0          0")

    lines += [
        "",
        "[XSECTIONS]",
        ";;Link           Shape        Geom1      Geom2      Geom3      Geom4      Barrels",
    ]
    for u, v, _ in edges:
        pipe_name = f"PIPE-{u}-{v}"
        diameter = min(BASE_PIPE_DIAMETER_M + DIAMETER_PER_UPSTREAM_M * counts[u], MAX_PIPE_DIAMETER_M)
        lines.append(f"{pipe_name:<16} CIRCULAR     {diameter:<10.2f} 0          0          0          1")
    for junction_name, outfall_name in outfall_targets.items():
        pipe_name = f"PIPE-{junction_name}-OUT"
        diameter = min(BASE_PIPE_DIAMETER_M + DIAMETER_PER_UPSTREAM_M * counts[junction_name], MAX_PIPE_DIAMETER_M)
        lines.append(f"{pipe_name:<16} CIRCULAR     {diameter:<10.2f} 0          0          0          1")

    return "\n".join(lines) + "\n"


def render_network_topology(junctions: dict[str, dict]) -> str:
    coords = ",\n".join(
        f'    "{name}": ({j["lon"]}, {j["lat"]})' for name, j in junctions.items()
    )
    return f'''"""Canonical lon/lat coordinates for the SWMM junctions in pune_base.inp.

pune_base.inp has no [COORDINATES] section, so this is the single source
of truth both the API's dry-network view and the terrain catchment
delineation key off of, instead of duplicating literals in each.

Generated by generate_network.py: {len(junctions)} real intersections from
routing.py's road graph, farthest-point sampled for spatial spread, each
connected in pune_base.inp to its nearest downhill neighbor.
"""

JUNCTION_COORDINATES: dict[str, tuple[float, float]] = {{
{coords},
}}
'''


if __name__ == "__main__":
    junctions, edges, outfall_targets = generate()
    print(f"{len(junctions)} junctions, {len(edges)} pipes, {len(outfall_targets)} outfall(s)")

    inp_text = render_inp(junctions, edges, outfall_targets)
    with open("pune_base.inp", "w") as f:
        f.write(inp_text)
    print("wrote pune_base.inp")

    topology_text = render_network_topology(junctions)
    with open("network_topology.py", "w") as f:
        f.write(topology_text)
    print("wrote network_topology.py")
