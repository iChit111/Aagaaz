"""Re-sizes pune_base.inp's pipe diameters from real accumulated demand.

generate_network.py's first pass sized pipes by a crude "diameter grows
with upstream junction count" heuristic, which ignores each junction's
actual catchment area, each pipe's real slope, and the storm's spatial
distribution -- so pipes ended up wildly under- or over-sized relative to
what actually reaches them, and 9 of 25 junctions were surcharging even
at 20mm/hr (light rain). This instead computes, per pipe, the combined
catchment area of everything upstream of it (terrain.py / runoff.py,
which need the network's junctions to already exist -- hence this being a
separate second pass rather than folded into generation), converts that
to a demand flow at a calibration rainfall using a flat design-storm
assumption (standard engineering practice for pipe sizing), and solves
Manning's equation for the diameter that exactly carries it. Below the
calibration rainfall the network should stay dry; above it, it should
increasingly surcharge.
"""

from __future__ import annotations

import math
import re

from network_topology import JUNCTION_COORDINATES
from runoff import CatchmentRunoff, RUNOFF_COEFFICIENT

INP_PATH = "pune_base.inp"

# Rainfall intensity (mm/hr) pipes are sized to just barely carry -- the
# network should start surcharging noticeably above this and stay dry below it.
CALIBRATION_RAINFALL_MM_HR = 60.0
MANNINGS_N = 0.013
MIN_DIAMETER_M = 0.2
MAX_DIAMETER_M = 1.5
MIN_SLOPE = 0.002  # floor to avoid absurd diameters on near-flat real segments


def _parse_inp(path: str):
    elevations: dict[str, float] = {}
    conduits: list[tuple[str, str, str, float]] = []  # (pipe_name, node1, node2, length_m)
    outfall_elevations: dict[str, float] = {}
    section = None
    with open(path) as f:
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
            elif section == "OUTFALLS":
                outfall_elevations[parts[0]] = float(parts[1])
            elif section == "CONDUITS":
                conduits.append((parts[0], parts[1], parts[2], float(parts[3])))
    return elevations, outfall_elevations, conduits


def _manning_diameter_m(demand_cms: float, slope: float) -> float:
    """Solve Manning's equation (circular, full-flow) for the diameter
    that exactly carries demand_cms at the given slope."""
    k = (math.pi / (4 * MANNINGS_N)) * (0.25 ** (2 / 3))  # Q = k * D^(8/3) * sqrt(S)
    slope = max(slope, MIN_SLOPE)
    diameter = (demand_cms / (k * math.sqrt(slope))) ** (3 / 8)
    return min(max(diameter, MIN_DIAMETER_M), MAX_DIAMETER_M)


def recalibrate() -> dict[str, float]:
    elevations, outfall_elevations, conduits = _parse_inp(INP_PATH)
    all_elevations = {**elevations, **outfall_elevations}

    # Build upstream-node sets: for each node, every node (including itself)
    # that drains through it.
    children: dict[str, list[str]] = {n: [] for n in all_elevations}
    for _pipe_name, u, v, _length in conduits:
        children[v].append(u)

    upstream_cache: dict[str, set[str]] = {}

    def upstream(node: str) -> set[str]:
        if node in upstream_cache:
            return upstream_cache[node]
        result = {node}
        for child in children.get(node, []):
            result |= upstream(child)
        upstream_cache[node] = result
        return result

    runoff = CatchmentRunoff()
    rainfall_m_per_s = CALIBRATION_RAINFALL_MM_HR / 1000.0 / 3600.0

    diameters: dict[str, float] = {}
    for pipe_name, u, v, length_m in conduits:
        upstream_nodes = upstream(u) & set(JUNCTION_COORDINATES)  # exclude synthetic OUTFALL nodes
        total_area_m2 = sum(runoff.catchment_area_m2(n) for n in upstream_nodes)
        demand_cms = rainfall_m_per_s * total_area_m2 * RUNOFF_COEFFICIENT

        drop_m = all_elevations[u] - all_elevations[v]
        slope = drop_m / length_m
        diameter = _manning_diameter_m(demand_cms, slope)
        diameters[pipe_name] = diameter

    return diameters


def rewrite_xsections(diameters: dict[str, float], path: str = INP_PATH) -> None:
    with open(path) as f:
        lines = f.readlines()

    out = []
    in_xsections = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_xsections = stripped.strip("[]") == "XSECTIONS"
            out.append(line)
            continue
        if in_xsections and stripped and not stripped.startswith(";"):
            pipe_name = stripped.split()[0]
            if pipe_name in diameters:
                out.append(f"{pipe_name:<16} CIRCULAR     {diameters[pipe_name]:<10.2f} 0          0          0          1\n")
                continue
        out.append(line)

    with open(path, "w") as f:
        f.writelines(out)


if __name__ == "__main__":
    diameters = recalibrate()
    for name, d in sorted(diameters.items()):
        print(f"{name}: {d:.2f}m")
    rewrite_xsections(diameters)
    print(f"Rewrote {len(diameters)} pipe diameters in {INP_PATH}")
