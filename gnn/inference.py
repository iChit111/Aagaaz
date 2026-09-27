"""GNN inference: build a live graph and predict surcharge volumes.

This module bridges the gap between a rainfall request and GNN predictions.
Unlike training (which reads pre-generated .npz files), inference builds the
PyG graph sequence on the fly from:
    - The current SWMM network topology (pune_base.inp)
    - Static node features (elevations, catchment areas)
    - Static edge features (pipe diameters, lengths, slopes, Manning's n)
    - Dynamic rainfall computed from nowcast.py's storm model

Usage:
    from gnn.inference import FloodGNNPredictor

    predictor = FloodGNNPredictor("checkpoints/flood_gnn_best.pt")
    frames = predictor.predict(rainfall_mm_hr=80.0)
    # frames is list[dict] with keys: elapsed_min, node_surcharges
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from gnn.model import FloodGNN

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CATCHMENTS_PATH = BASE_DIR / "data" / "catchments.npz"
CELL_AREA_M2 = 900.0

# Timesteps matching generate_training_data.py's 15-min intervals over 3 hours
ELAPSED_MINUTES = list(range(0, 181, 15))  # [0, 15, 30, ..., 180]


# ---------------------------------------------------------------------------
# Helpers (mirror graph_builder.py's feature construction)
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
    storm_offset: tuple[float, float] = (0.0, 0.0),
    velocity_deg_per_min: tuple[float, float] = (0.00003, -0.00002),
    radius_m: float = 1800.0,
    peak_at_min: float = 90.0,
    spread_min: float = 60.0,
) -> float:
    cx = center0[0] + storm_offset[0] + velocity_deg_per_min[0] * elapsed_min
    cy = center0[1] + storm_offset[1] + velocity_deg_per_min[1] * elapsed_min
    dist_m = _haversine_m(lon, lat, cx, cy)
    spatial = math.exp(-(dist_m ** 2) / (2 * radius_m ** 2))
    temporal = math.exp(-((elapsed_min - peak_at_min) / spread_min) ** 2)
    return peak_mm_hr * spatial * temporal


def _load_catchment_areas(path: Path = CATCHMENTS_PATH) -> dict[str, float]:
    areas: dict[str, float] = {}
    try:
        with np.load(path) as data:
            node_ids = sorted({k.rsplit("__", 1)[0] for k in data.files})
            for nid in node_ids:
                mask = data[f"{nid}__mask"]
                areas[nid] = float(mask.sum()) * CELL_AREA_M2
    except FileNotFoundError:
        pass
    return areas


def _parse_network(inp_path: str) -> dict:
    """Extract junction elevations and conduit properties from pune_base.inp.

    Returns dict with keys:
        - elevations: {node_id: elevation_m}
        - conduits: list of {pipe_id, from_node, to_node, length_m}
        - xsections: {pipe_id: {shape, geom1 (diameter), ...}}
    """
    elevations: dict[str, float] = {}
    conduits: list[dict] = []
    xsections: dict[str, dict] = {}
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
            if section == "JUNCTIONS" and len(parts) >= 2:
                elevations[parts[0]] = float(parts[1])
            elif section == "CONDUITS" and len(parts) >= 5:
                conduits.append({
                    "pipe_id": parts[0],
                    "from_node": parts[1],
                    "to_node": parts[2],
                    "length_m": float(parts[3]),
                })
            elif section == "XSECTIONS" and len(parts) >= 3:
                xsections[parts[0]] = {
                    "shape": parts[1],
                    "geom1": float(parts[2]),  # diameter for CIRCULAR
                }

    return {
        "elevations": elevations,
        "conduits": conduits,
        "xsections": xsections,
    }


# ---------------------------------------------------------------------------
# FloodGNNPredictor
# ---------------------------------------------------------------------------

class FloodGNNPredictor:
    """Loads a trained FloodGNN checkpoint and predicts surcharge volumes.

    Call .predict(rainfall_mm_hr) to get per-node surcharge volumes at
    each 15-minute timestep over 3 hours.

    Caches all static data (network topology, catchments, graph structure)
    at construction time — only the rainfall features change per request.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
    ):
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        self._model = FloodGNN(
            n_node_feat=ckpt.get("n_node_feat", 4),
            n_edge_feat=ckpt.get("n_edge_feat", 4),
            hidden_dim=ckpt.get("hidden_dim", 128),
            n_mp_layers=ckpt.get("n_mp_layers", 3),
            dropout=ckpt.get("dropout", 0.1),
            log_targets=ckpt.get("log_targets", True),
        ).to(self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        logger.info(
            "Loaded FloodGNN from epoch %d (val_loss=%.6f) on %s",
            ckpt.get("epoch", -1),
            ckpt.get("best_val_loss", float("nan")),
            self._device,
        )

        # Load static network data
        from network_topology import JUNCTION_COORDINATES
        self._junction_coords = JUNCTION_COORDINATES
        self._node_ids = sorted(JUNCTION_COORDINATES.keys())
        self._N = len(self._node_ids)

        # Junction center for rainfall computation
        lons = [c[0] for c in JUNCTION_COORDINATES.values()]
        lats = [c[1] for c in JUNCTION_COORDINATES.values()]
        self._center = (sum(lons) / len(lons), sum(lats) / len(lats))

        # Catchment areas
        catchment_areas_map = _load_catchment_areas()
        catchment_areas = np.array(
            [catchment_areas_map.get(nid, 0.0) for nid in self._node_ids],
            dtype=np.float32,
        )
        self._catchment_log = np.log1p(catchment_areas)

        # Elevations
        inp_path = str(BASE_DIR / "pune_base.inp")
        network = _parse_network(inp_path)
        node_elevations = np.array(
            [network["elevations"].get(nid, 0.0) for nid in self._node_ids],
            dtype=np.float32,
        )
        elev_mean = node_elevations.mean()
        elev_std = max(node_elevations.std(), 1.0)
        self._elev_norm = (node_elevations - elev_mean) / elev_std

        # Build edge_index and edge_attr from conduit topology
        node_to_idx = {nid: j for j, nid in enumerate(self._node_ids)}
        valid_conduits = [
            c for c in network["conduits"]
            if c["from_node"] in node_to_idx and c["to_node"] in node_to_idx
        ]

        edge_from = [node_to_idx[c["from_node"]] for c in valid_conduits]
        edge_to = [node_to_idx[c["to_node"]] for c in valid_conduits]
        self._edge_index = torch.tensor(
            [edge_from, edge_to], dtype=torch.long, device=self._device,
        )

        # Edge features: diameter, log(length), slope, Manning's n
        # And compute per-node derived capacity features simultaneously
        edge_feats = []
        min_pipe_diam = np.full(self._N, 1.0, dtype=np.float32)
        total_pipe_capacity = np.zeros(self._N, dtype=np.float32)
        
        for c in valid_conduits:
            xs = network["xsections"].get(c["pipe_id"], {})
            diameter = xs.get("geom1", 0.5)  # default 0.5m
            length = c["length_m"]
            from_elev = network["elevations"].get(c["from_node"], 0.0)
            to_elev = network["elevations"].get(c["to_node"], 0.0)
            slope = abs(from_elev - to_elev) / max(length, 0.1)
            mannings_n = 0.013  # concrete default
            
            capacity = (diameter ** (8.0 / 3.0)) / mannings_n * (max(slope, 1e-4) ** 0.5)
            src_idx = node_to_idx[c["from_node"]]
            dst_idx = node_to_idx[c["to_node"]]
            for node_idx in (src_idx, dst_idx):
                min_pipe_diam[node_idx] = min(min_pipe_diam[node_idx], diameter)
                total_pipe_capacity[node_idx] += capacity
                
            edge_feats.append([diameter, math.log1p(length), slope, mannings_n])

        self._edge_attr = torch.tensor(
            edge_feats, dtype=torch.float32, device=self._device,
        )
        self._min_pipe_diam = min_pipe_diam
        self._pipe_capacity_log = np.log1p(total_pipe_capacity)

        logger.info(
            "Inference ready: %d nodes, %d edges, %d timesteps",
            self._N, self._edge_index.shape[1], len(ELAPSED_MINUTES),
        )

    def predict(
        self,
        rainfall_mm_hr: float,
        storm_offset: tuple[float, float] = (0.0, 0.0),
    ) -> list[dict]:
        """Predict per-node surcharge volumes for a given rainfall intensity.

        Args:
            rainfall_mm_hr: Peak rainfall intensity (mm/hr).
            storm_offset: (lon, lat) offset from junction center for storm position.

        Returns:
            List of dicts, one per timestep:
                {
                    "elapsed_min": int,
                    "node_surcharges": {node_id: surcharge_volume_m3}
                }
        """
        # Build the PyG sequence on the fly
        sequence: list[Data] = []
        cumulative_rainfall = np.zeros(self._N, dtype=np.float32)

        for t_idx, t_min in enumerate(ELAPSED_MINUTES):
            # Per-node rainfall intensity at this timestep
            rainfall = np.array([
                _compute_rainfall(
                    self._junction_coords[nid][0],
                    self._junction_coords[nid][1],
                    float(t_min),
                    rainfall_mm_hr,
                    self._center,
                    storm_offset,
                )
                for nid in self._node_ids
            ], dtype=np.float32)

            # Accumulate rainfall
            if t_idx > 0:
                dt_hr = (t_min - ELAPSED_MINUTES[t_idx - 1]) / 60.0
                cumulative_rainfall += rainfall * dt_hr

            # Node features: (N, 6)
            x = torch.tensor(
                np.column_stack([
                    self._elev_norm,
                    self._catchment_log,
                    rainfall,
                    cumulative_rainfall,
                    self._min_pipe_diam,
                    self._pipe_capacity_log,
                ]),
                dtype=torch.float32,
                device=self._device,
            )

            # Dummy target (not used during inference)
            y = torch.zeros(self._N, dtype=torch.float32, device=self._device)

            sequence.append(Data(
                x=x,
                edge_index=self._edge_index,
                edge_attr=self._edge_attr,
                y=y,
            ))

        # Run inference
        preds = self._model.predict(sequence)  # (T, N)

        # Build response
        frames = []
        for t_idx, t_min in enumerate(ELAPSED_MINUTES):
            surcharges = {}
            for j, nid in enumerate(self._node_ids):
                vol = float(preds[t_idx, j].item())
                if vol > 0.01:  # filter noise
                    surcharges[nid] = vol
            frames.append({
                "elapsed_min": t_min,
                "node_surcharges": surcharges,
            })

        return frames
