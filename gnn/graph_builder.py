"""Build PyTorch Geometric graphs from .npz training data and static network files.

This module is the bridge between generate_training_data.py's raw SWMM
output and the GNN model's input.  It constructs two related graphs:

1. **Drainage graph** — SWMM junctions as nodes, conduits as directed edges.
   This is where the GNN learns surcharge prediction.

2. **Road graph** — road intersections as nodes, road segments as edges.
   Used only during inference-time post-processing to map predicted
   junction surcharges onto per-road flood depths (via flood_fill.py,
   kept as a post-processing step per the migration plan).

The drainage graph is what gets batched and fed to the model.  Each
graph snapshot carries:
    - Static node features:   elevation, catchment area
    - Static edge features:   pipe diameter, length, slope, Manning's n
    - Dynamic node features:  rainfall intensity at that timestep
    - Target node labels:     surcharge volume at each junction

Temporal unrolling (the "full 3-hr time series" requirement) is handled
by representing each 15-min snapshot as a separate Data object within a
temporal sequence, grouped by scenario_id so the training loop can feed
them sequentially into the GNN+GRU.

Usage:
    from gnn.graph_builder import DrainageGraphBuilder, TrainingDataset

    builder = DrainageGraphBuilder()
    dataset = TrainingDataset("data/gnn_training", builder)
    seq = dataset[0]  # list[Data] — one Data per timestep in the scenario
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError(
        "PyTorch Geometric is required.  Install with:\n"
        "  pip install torch-geometric\n"
        "See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    )

BASE_DIR = Path(__file__).resolve().parent.parent
ROADS_PATH = BASE_DIR / "frontend" / "src" / "pune_roads.json"
CATCHMENTS_PATH = BASE_DIR / "data" / "catchments.npz"

# SRTM cell area in m^2, matches terrain.py / runoff.py
CELL_AREA_M2 = 900.0


# ---------------------------------------------------------------------------
# Catchment areas (static, computed once from terrain.py output)
# ---------------------------------------------------------------------------

def _load_catchment_areas(path: Path = CATCHMENTS_PATH) -> dict[str, float]:
    """Return {node_id: catchment_area_m2} from the terrain.py .npz."""
    areas: dict[str, float] = {}
    try:
        with np.load(path) as data:
            node_ids = sorted({k.rsplit("__", 1)[0] for k in data.files})
            for nid in node_ids:
                mask = data[f"{nid}__mask"]
                areas[nid] = float(mask.sum()) * CELL_AREA_M2
    except FileNotFoundError:
        pass  # Catchment areas will be zero — logged as warning at build time
    return areas


# ---------------------------------------------------------------------------
# Rainfall computation (reproduces nowcast.py's intensity_mm_per_hr)
# ---------------------------------------------------------------------------

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def _compute_rainfall(
    lon: float,
    lat: float,
    elapsed_min: float,
    peak_mm_hr: float,
    center0: tuple[float, float],
    storm_offset: tuple[float, float],
    velocity_deg_per_min: tuple[float, float] = (0.00003, -0.00002),
    radius_m: float = 1800.0,
    peak_at_min: float = 90.0,
    spread_min: float = 60.0,
) -> float:
    """Reproduce nowcast.py's synthetic storm intensity at (lon, lat, t)."""
    cx = center0[0] + storm_offset[0] + velocity_deg_per_min[0] * elapsed_min
    cy = center0[1] + storm_offset[1] + velocity_deg_per_min[1] * elapsed_min
    dist_m = _haversine_m(lon, lat, cx, cy)
    spatial = math.exp(-(dist_m ** 2) / (2 * radius_m ** 2))
    temporal = math.exp(-((elapsed_min - peak_at_min) / spread_min) ** 2)
    return peak_mm_hr * spatial * temporal


# ---------------------------------------------------------------------------
# DrainageGraphBuilder
# ---------------------------------------------------------------------------

class DrainageGraphBuilder:
    """Constructs PyG Data objects from per-scenario .npz files.

    Caches static data (catchment areas, junction coordinates) and builds
    one Data object per timestep per scenario.
    """

    def __init__(
        self,
        catchments_path: Path = CATCHMENTS_PATH,
    ):
        self._catchment_areas = _load_catchment_areas(catchments_path)

        # Junction center (for rainfall computation) — computed lazily
        # from the first scenario's node coordinates.
        self._junction_center: Optional[tuple[float, float]] = None

    def _get_junction_center(self, junction_coords: dict[str, tuple[float, float]]) -> tuple[float, float]:
        if self._junction_center is None:
            lons = [c[0] for c in junction_coords.values()]
            lats = [c[1] for c in junction_coords.values()]
            self._junction_center = (sum(lons) / len(lons), sum(lats) / len(lats))
        return self._junction_center

    def build_sequence(
        self,
        npz_path: Path,
        junction_coords: dict[str, tuple[float, float]],
    ) -> list[Data]:
        """Load a scenario .npz and return a list of Data objects (one per timestep).

        Each Data object contains:
            x:              (N, F_node) node feature matrix
            edge_index:     (2, E) directed drainage edges
            edge_attr:      (E, F_edge) edge feature matrix
            y:              (N,) target surcharge volume per node
            elapsed_min:    scalar — timestep in the simulation

        Node features (F_node = 6):
            0: elevation (m), normalised
            1: catchment area (m^2), log-scaled
            2: rainfall intensity (mm/hr) at this timestep
            3: cumulative rainfall (mm) up to this timestep
            4: min pipe diameter (m) of connected pipes
            5: total pipe capacity (Manning's proxy), log-scaled

        Edge features (F_edge = 4):
            0: pipe diameter (m)
            1: pipe length (m), log-scaled
            2: pipe slope
            3: Manning's n
        """
        with np.load(npz_path, allow_pickle=False) as data:
            node_ids = list(data["node_ids"])
            node_elevations = data["node_elevations"]          # (N,)
            edge_index = torch.from_numpy(data["edge_index"])  # (2, E)
            pipe_diameters = data["pipe_diameters"]            # (E,)
            pipe_lengths = data["pipe_lengths"]                # (E,)
            pipe_slopes = data["pipe_slopes"]                  # (E,)
            pipe_mannings = data["pipe_mannings"]              # (E,)
            elapsed_minutes = data["elapsed_minutes"]          # (T,)
            surcharge_volumes = data["surcharge_volumes"]      # (T, N)
            rainfall_mm_hr = float(data["rainfall_mm_hr"])
            storm_offset = tuple(data["storm_offset"].tolist())

        N = len(node_ids)
        E = edge_index.shape[1]
        center = self._get_junction_center(junction_coords)

        # Static node features
        elev_mean = node_elevations.mean()
        elev_std = max(node_elevations.std(), 1.0)
        elev_norm = (node_elevations - elev_mean) / elev_std

        catchment_areas = np.array(
            [self._catchment_areas.get(nid, 0.0) for nid in node_ids],
            dtype=np.float32,
        )
        catchment_log = np.log1p(catchment_areas)

        # Derived node features: per-node pipe capacity indicators
        # These tell the model the flooding threshold at each junction
        # without needing to discover it through message passing.
        min_pipe_diam = np.full(N, 1.0, dtype=np.float32)  # default 1m if no pipes
        total_pipe_capacity = np.zeros(N, dtype=np.float32)
        for e in range(E):
            src, dst = int(edge_index[0, e]), int(edge_index[1, e])
            d = float(pipe_diameters[e])
            # Manning's pipe capacity proxy: D^(8/3) / n * sqrt(S)
            n = float(pipe_mannings[e])
            s = max(float(pipe_slopes[e]), 1e-4)
            capacity = (d ** (8.0 / 3.0)) / n * (s ** 0.5)
            for node_idx in (src, dst):
                min_pipe_diam[node_idx] = min(min_pipe_diam[node_idx], d)
                total_pipe_capacity[node_idx] += capacity

        # Static edge features
        edge_attr = torch.tensor(
            np.column_stack([
                pipe_diameters,
                np.log1p(pipe_lengths),
                pipe_slopes,
                pipe_mannings,
            ]),
            dtype=torch.float32,
        )

        # Build one Data per timestep
        sequence: list[Data] = []
        cumulative_rainfall = np.zeros(N, dtype=np.float32)

        for t_idx, t_min in enumerate(elapsed_minutes):
            # Compute per-node rainfall at this timestep
            rainfall = np.array([
                _compute_rainfall(
                    junction_coords[nid][0],
                    junction_coords[nid][1],
                    float(t_min),
                    rainfall_mm_hr,
                    center,
                    storm_offset,
                )
                for nid in node_ids
            ], dtype=np.float32)

            # Accumulate rainfall (trapezoidal: intensity × dt)
            if t_idx > 0:
                dt_hr = (t_min - elapsed_minutes[t_idx - 1]) / 60.0
                cumulative_rainfall += rainfall * dt_hr  # mm

            # Node feature matrix: (N, 6)
            x = torch.tensor(
                np.column_stack([
                    elev_norm,
                    catchment_log,
                    rainfall,
                    cumulative_rainfall,
                    min_pipe_diam,
                    np.log1p(total_pipe_capacity),
                ]),
                dtype=torch.float32,
            )

            # Target: per-node surcharge volume
            y = torch.tensor(surcharge_volumes[t_idx], dtype=torch.float32)

            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                elapsed_min=torch.tensor(float(t_min)),
            )
            # Store metadata for downstream use (not batched by PyG)
            graph.node_ids = node_ids
            graph.rainfall_mm_hr = rainfall_mm_hr
            graph.storm_offset = storm_offset

            sequence.append(graph)

        return sequence


# ---------------------------------------------------------------------------
# TrainingDataset — PyTorch Dataset over scenario .npz files
# ---------------------------------------------------------------------------

class TrainingDataset(Dataset):
    """A Dataset that yields temporal sequences of PyG Data objects.

    Each item is a list[Data] of length T (one per 15-min snapshot),
    representing a full 3-hour simulation scenario.

    Usage:
        dataset = TrainingDataset("data/gnn_training")
        for seq in DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0]):
            # seq is list[Data], len(seq) == T
            ...
    """

    def __init__(
        self,
        data_dir: str | Path,
        builder: Optional[DrainageGraphBuilder] = None,
        junction_coords: Optional[dict[str, tuple[float, float]]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.builder = builder or DrainageGraphBuilder()

        # Load junction coordinates
        if junction_coords is not None:
            self._junction_coords = junction_coords
        else:
            # Import from the project's network topology
            import sys
            sys.path.insert(0, str(BASE_DIR))
            from network_topology import JUNCTION_COORDINATES
            self._junction_coords = JUNCTION_COORDINATES

        # Discover scenario files
        self._files = sorted(self.data_dir.glob("*.npz"))
        if not self._files:
            raise FileNotFoundError(
                f"No .npz scenario files found in {self.data_dir}. "
                "Run generate_training_data.py first."
            )

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> list[Data]:
        return self.builder.build_sequence(
            self._files[idx],
            self._junction_coords,
        )

    @property
    def scenario_ids(self) -> list[str]:
        return [f.stem for f in self._files]

    @property
    def n_node_features(self) -> int:
        """Number of node features (F_node) in each Data.x."""
        return 6  # elevation, catchment_area, rainfall, cumulative_rainfall, min_pipe_diam, pipe_capacity

    @property
    def n_edge_features(self) -> int:
        """Number of edge features (F_edge) in each Data.edge_attr."""
        return 4  # diameter, length, slope, mannings_n


# ---------------------------------------------------------------------------
# Utility: split dataset into train / val / test
# ---------------------------------------------------------------------------

def train_val_test_split(
    dataset: TrainingDataset,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Return (train_indices, val_indices, test_indices) for the dataset.

    Splits are stratified by rainfall intensity so each split sees all
    intensity levels, preventing the model from never training on
    extreme events.
    """
    rng = np.random.default_rng(seed)

    # Group indices by rainfall intensity (encoded in the filename)
    from collections import defaultdict
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, fname in enumerate(dataset._files):
        # Filename format: r{rain}_o{...}_p{...}.npz
        rain_key = fname.stem.split("_")[0]  # e.g. "r50"
        groups[rain_key].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for _key, indices in sorted(groups.items()):
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# CLI: quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO)

    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(BASE_DIR / "data" / "gnn_training")

    try:
        dataset = TrainingDataset(data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Dataset: {len(dataset)} scenarios in {data_dir}")
    print(f"Node features: {dataset.n_node_features}")
    print(f"Edge features: {dataset.n_edge_features}")

    # Load and inspect the first scenario
    seq = dataset[0]
    print(f"\nFirst scenario: {len(seq)} timesteps")
    for i, g in enumerate(seq):
        print(
            f"  t={g.elapsed_min.item():>5.0f}min  "
            f"nodes={g.x.shape[0]}  edges={g.edge_index.shape[1]}  "
            f"x={list(g.x.shape)}  edge_attr={list(g.edge_attr.shape)}  "
            f"y_max={g.y.max().item():.2f} m^3"
        )

    # Show split sizes
    train, val, test = train_val_test_split(dataset)
    print(f"\nSplit: train={len(train)} val={len(val)} test={len(test)}")
