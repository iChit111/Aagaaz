"""Generate training data for the flood nowcasting GNN.

Runs the existing PySWMM pipeline across a grid of:
    rainfall intensities × storm positions × pipe-parameter perturbations

and records, at each 15-minute snapshot, the per-junction surcharge volume
and per-road flood depth ground truth.  Output is a directory of .npz files
that gnn/graph_builder.py can load into PyTorch Geometric Data objects.

Why the perturbation layer:
    pune_base.inp's pipe diameters are estimated, not surveyed.  By training
    the GNN on many plausible pipe configurations (±20% diameter, variable
    Manning's n), it learns physics robust to inaccurate dimensions rather
    than overfitting to one specific (likely wrong) network.

Usage:
    python generate_training_data.py [--out-dir data/gnn_training] [--workers 4]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
INP_PATH = BASE_DIR / "pune_base.inp"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "gnn_training"

# ---------------------------------------------------------------------------
# Scenario grid
# ---------------------------------------------------------------------------
RAINFALL_INTENSITIES = [10, 20, 30, 50, 75, 100, 150]  # mm/hr
STORM_OFFSETS_DEG = [
    (0.0, 0.0),          # centered on network
    (+0.005, 0.0),       # east
    (-0.005, 0.0),       # west
    (0.0, +0.004),       # north
    (0.0, -0.004),       # south
]
N_PERTURBATION_SAMPLES = 10
DIAMETER_NOISE_RANGE = (0.80, 1.20)       # ±20%
MANNING_N_RANGE = (0.010, 0.016)          # concrete to rough concrete

REPORT_INTERVAL_MIN = 15
SIMULATION_DURATION_MIN = 180  # 3 hours


# ---------------------------------------------------------------------------
# .inp perturbation
# ---------------------------------------------------------------------------

def _perturb_inp(
    src_path: Path,
    dest_path: Path,
    rng: random.Random,
) -> dict[str, float]:
    """Write a perturbed copy of the .inp, returning {pipe_id: new_diameter_m}.

    Randomises per-pipe:
        - diameter: scaled by a uniform factor in DIAMETER_NOISE_RANGE
        - Manning's n: drawn uniformly from MANNING_N_RANGE

    All other sections are left untouched.
    """
    diameter_factors: dict[str, float] = {}
    mannings: dict[str, float] = {}

    with open(src_path) as f:
        lines = f.readlines()

    out: list[str] = []
    section: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped.strip("[]")
            out.append(line)
            continue

        if section == "XSECTIONS" and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            pipe_id = parts[0]
            old_diam = float(parts[2])
            factor = rng.uniform(*DIAMETER_NOISE_RANGE)
            new_diam = old_diam * factor
            diameter_factors[pipe_id] = new_diam
            out.append(
                f"{pipe_id:<16} {parts[1]:<12} {new_diam:<10.4f}"
                f" 0          0          0          1\n"
            )
            continue

        if section == "CONDUITS" and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            pipe_id = parts[0]
            n = rng.uniform(*MANNING_N_RANGE)
            mannings[pipe_id] = n
            parts[4] = f"{n:.4f}"
            out.append(f"{parts[0]:<16} {parts[1]:<16} {parts[2]:<16} "
                       f"{parts[3]:<10} {parts[4]:<10} "
                       f"{parts[5]:<10} {parts[6]:<10} "
                       f"{parts[7]:<10} {parts[8]}\n")
            continue

        out.append(line)

    with open(dest_path, "w") as f:
        f.writelines(out)

    return diameter_factors


# ---------------------------------------------------------------------------
# Parse static network attributes from .inp
# ---------------------------------------------------------------------------

def parse_network(inp_path: Path) -> dict:
    """Extract junction elevations, conduit topology, diameters, roughness.

    Returns a dict consumable by graph_builder as static network features.
    """
    elevations: dict[str, float] = {}
    outfall_elevations: dict[str, float] = {}
    conduits: list[dict] = []
    xsections: dict[str, dict] = {}
    section: str | None = None

    with open(inp_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if stripped.startswith("["):
                section = stripped.strip("[]")
                continue
            parts = stripped.split()
            if section == "JUNCTIONS":
                elevations[parts[0]] = float(parts[1])
            elif section == "OUTFALLS":
                outfall_elevations[parts[0]] = float(parts[1])
            elif section == "CONDUITS":
                conduits.append({
                    "pipe_id": parts[0],
                    "from_node": parts[1],
                    "to_node": parts[2],
                    "length_m": float(parts[3]),
                    "mannings_n": float(parts[4]),
                })
            elif section == "XSECTIONS":
                xsections[parts[0]] = {
                    "shape": parts[1],
                    "diameter_m": float(parts[2]),
                }

    # Merge xsection data into conduits
    for c in conduits:
        xs = xsections.get(c["pipe_id"], {})
        c["diameter_m"] = xs.get("diameter_m", 0.3)
        c["shape"] = xs.get("shape", "CIRCULAR")

    all_elevations = {**elevations, **outfall_elevations}
    for c in conduits:
        c["slope"] = max(
            (all_elevations.get(c["from_node"], 0) - all_elevations.get(c["to_node"], 0))
            / max(c["length_m"], 1.0),
            0.001,
        )

    return {
        "elevations": elevations,
        "outfall_elevations": outfall_elevations,
        "conduits": conduits,
    }


# ---------------------------------------------------------------------------
# Run one SWMM simulation and capture time series
# ---------------------------------------------------------------------------

def _run_single_simulation(
    inp_path: Path,
    rainfall_mm_hr: float,
    storm_offset_deg: tuple[float, float],
) -> list[dict]:
    """Run PySWMM for a single scenario, returning a list of snapshot dicts.

    Each snapshot: {
        "elapsed_min": int,
        "surcharge_volumes": {node_id: float},  # m^3
        "road_depths": {road_id: float},         # cm
    }

    Imports are done inside the function so this can be called in a
    subprocess pool without loading everything at module import time.
    """
    from pyswmm import Simulation, Nodes
    from network_topology import JUNCTION_COORDINATES
    from nowcast import DEFAULT_STORM, StormCell
    from runoff import CatchmentRunoff
    from flood_fill import FloodExtentEstimator

    # Shift storm center by offset
    storm = replace(
        DEFAULT_STORM,
        peak_intensity_mm_hr=rainfall_mm_hr,
        center0=(
            DEFAULT_STORM.center0[0] + storm_offset_deg[0],
            DEFAULT_STORM.center0[1] + storm_offset_deg[1],
        ),
    )

    runoff = CatchmentRunoff()
    flood_est = FloodExtentEstimator()
    report_interval = timedelta(minutes=REPORT_INTERVAL_MIN)

    snapshots: list[dict] = []

    with Simulation(str(inp_path)) as sim:
        nodes = {nid: Nodes(sim)[nid] for nid in JUNCTION_COORDINATES}
        start_time = sim.start_time
        next_report = start_time

        for _step in sim:
            elapsed_min = (sim.current_time - start_time).total_seconds() / 60
            for nid, node in nodes.items():
                node.generated_inflow(runoff.inflow_cms(nid, elapsed_min, storm))

            if not snapshots:
                snapshots.append(_capture_snapshot(0, nodes, flood_est))

            if sim.current_time >= next_report + report_interval:
                next_report += report_interval
                t = int((next_report - start_time).total_seconds() // 60)
                snapshots.append(_capture_snapshot(t, nodes, flood_est))

        final_t = int((sim.current_time - start_time).total_seconds() // 60)
        if final_t != snapshots[-1]["elapsed_min"]:
            snapshots.append(_capture_snapshot(final_t, nodes, flood_est))

    return snapshots


def _capture_snapshot(
    elapsed_min: int,
    nodes: dict,
    flood_est,
) -> dict:
    """Capture per-node surcharge volumes and per-road flood depths."""
    surcharge_volumes: dict[str, float] = {}
    surcharging_nodes = []

    for nid, node in nodes.items():
        vol = node.statistics["flooding_volume"]
        surcharge_volumes[nid] = vol
        surcharging_nodes.append({
            "node_id": nid,
            "surcharge_volume_m3": vol,
        })

    # Compute road depths via the bathtub-fill pipeline
    cell_depth_cm: dict[tuple[int, int], float] = {}
    for sn in surcharging_nodes:
        extent = flood_est.flood_extent_by_index(sn["node_id"], sn["surcharge_volume_m3"])
        for idx, depth_cm in extent.items():
            if depth_cm > cell_depth_cm.get(idx, 0.0):
                cell_depth_cm[idx] = depth_cm

    road_depths = flood_est.road_depths(cell_depth_cm) if cell_depth_cm else {}

    return {
        "elapsed_min": elapsed_min,
        "surcharge_volumes": surcharge_volumes,
        "road_depths": road_depths,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_dataset(
    out_dir: Path = DEFAULT_OUT_DIR,
    inp_path: Path = INP_PATH,
    max_scenarios: int | None = None,
) -> int:
    """Generate the full training dataset.

    Each scenario is saved as a separate .npz file containing:
        - "rainfall_mm_hr": scalar
        - "storm_offset": (lon_offset, lat_offset)
        - "network": serialised network dict (elevations, conduits, diameters)
        - "snapshots": list of (elapsed_min, node_surcharges, road_depths)

    Returns the number of scenarios successfully generated.
    """
    from network_topology import JUNCTION_COORDINATES

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute the scenario grid
    scenarios: list[dict] = []
    for rain in RAINFALL_INTENSITIES:
        for offset in STORM_OFFSETS_DEG:
            for pert_idx in range(N_PERTURBATION_SAMPLES):
                scenarios.append({
                    "rainfall_mm_hr": rain,
                    "storm_offset": offset,
                    "perturbation_idx": pert_idx,
                })

    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]

    logger.info(
        "Generating %d scenarios (%d rainfall × %d offsets × %d perturbations)",
        len(scenarios), len(RAINFALL_INTENSITIES),
        len(STORM_OFFSETS_DEG), N_PERTURBATION_SAMPLES,
    )

    node_ids = sorted(JUNCTION_COORDINATES.keys())
    n_success = 0
    t0 = time.time()

    for i, scenario in enumerate(scenarios):
        scenario_id = (
            f"r{scenario['rainfall_mm_hr']}"
            f"_o{scenario['storm_offset'][0]:+.4f}_{scenario['storm_offset'][1]:+.4f}"
            f"_p{scenario['perturbation_idx']:02d}"
        )
        out_file = out_dir / f"{scenario_id}.npz"

        if out_file.exists():
            logger.debug("Skipping existing %s", scenario_id)
            n_success += 1
            continue

        # Create a perturbed .inp in a temp directory
        rng = random.Random(42 + i)
        tmp_dir = Path(tempfile.mkdtemp(prefix="gnn_train_"))
        perturbed_inp = tmp_dir / "model.inp"

        try:
            diameters = _perturb_inp(inp_path, perturbed_inp, rng)
            network = parse_network(perturbed_inp)

            snapshots = _run_single_simulation(
                perturbed_inp,
                scenario["rainfall_mm_hr"],
                scenario["storm_offset"],
            )

            # Serialise to .npz
            # Node surcharge time series: shape (T, N_nodes) — ordered by node_ids
            T = len(snapshots)
            surcharge_matrix = np.zeros((T, len(node_ids)), dtype=np.float32)
            elapsed_minutes = np.zeros(T, dtype=np.float32)

            for t_idx, snap in enumerate(snapshots):
                elapsed_minutes[t_idx] = snap["elapsed_min"]
                for j, nid in enumerate(node_ids):
                    surcharge_matrix[t_idx, j] = snap["surcharge_volumes"].get(nid, 0.0)

            # Road depths: use canonical road list so shape is consistent
            # across all scenarios (including dry ones with no flooding).
            if not hasattr(generate_dataset, '_canonical_road_ids'):
                import json as _json
                with open(BASE_DIR / "frontend" / "src" / "pune_roads.json") as _rf:
                    _roads = _json.load(_rf)
                generate_dataset._canonical_road_ids = sorted(
                    f.get("id") or f.get("properties", {}).get("@id", "")
                    for f in _roads.get("features", [])
                    if f.get("id") or f.get("properties", {}).get("@id")
                )
            road_ids = generate_dataset._canonical_road_ids

            road_depth_matrix = np.zeros((T, len(road_ids)), dtype=np.float32)
            road_id_to_idx = {rid: r_idx for r_idx, rid in enumerate(road_ids)}
            for t_idx, snap in enumerate(snapshots):
                for rid, depth in snap["road_depths"].items():
                    if rid in road_id_to_idx:
                        road_depth_matrix[t_idx, road_id_to_idx[rid]] = depth

            # Pipe topology and static edge features
            node_to_idx = {nid: j for j, nid in enumerate(node_ids)}
            valid_conduits = [
                c for c in network["conduits"]
                if c["from_node"] in node_to_idx and c["to_node"] in node_to_idx
            ]
            pipe_ids = [c["pipe_id"] for c in valid_conduits]
            pipe_diameters = np.array(
                [c["diameter_m"] for c in valid_conduits], dtype=np.float32
            )
            pipe_lengths = np.array(
                [c["length_m"] for c in valid_conduits], dtype=np.float32
            )
            pipe_mannings = np.array(
                [c["mannings_n"] for c in valid_conduits], dtype=np.float32
            )
            pipe_slopes = np.array(
                [c["slope"] for c in valid_conduits], dtype=np.float32
            )

            edge_from = [node_to_idx[c["from_node"]] for c in valid_conduits]
            edge_to = [node_to_idx[c["to_node"]] for c in valid_conduits]
            edge_index = np.array([edge_from, edge_to], dtype=np.int64)

            # Junction elevations (ordered by node_ids)
            node_elevations = np.array(
                [network["elevations"].get(nid, 0.0) for nid in node_ids],
                dtype=np.float32,
            )

            np.savez_compressed(
                out_file,
                # Scenario metadata
                rainfall_mm_hr=np.float32(scenario["rainfall_mm_hr"]),
                storm_offset=np.array(scenario["storm_offset"], dtype=np.float32),
                # Time series targets
                elapsed_minutes=elapsed_minutes,
                surcharge_volumes=surcharge_matrix,       # (T, N_nodes)
                road_depths=road_depth_matrix,            # (T, N_roads)
                # Static network
                node_ids=np.array(node_ids),
                node_elevations=node_elevations,
                edge_index=edge_index,                    # (2, N_edges)
                pipe_diameters=pipe_diameters,
                pipe_lengths=pipe_lengths,
                pipe_mannings=pipe_mannings,
                pipe_slopes=pipe_slopes,
                pipe_ids=np.array(pipe_ids),
                road_ids=np.array(road_ids),
            )

            n_success += 1
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(scenarios) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    "[%4d/%d] %s  (%.1f scen/s, ETA %.0fs)",
                    i + 1, len(scenarios), scenario_id, rate, eta,
                )

        except Exception:
            logger.exception("Failed scenario %s", scenario_id)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - t0
    logger.info(
        "Done: %d/%d scenarios in %.1fs (%.1f scen/s)",
        n_success, len(scenarios), elapsed,
        n_success / elapsed if elapsed > 0 else 0,
    )
    return n_success


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Generate GNN training data from SWMM simulations.")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--max-scenarios", type=int, default=None,
        help="Cap the number of scenarios (useful for testing).",
    )
    args = parser.parse_args()

    n = generate_dataset(out_dir=args.out_dir, max_scenarios=args.max_scenarios)
    print(f"\n{n} scenario file(s) written to {args.out_dir}")
