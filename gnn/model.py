"""Spatio-temporal GNN for drainage network surcharge prediction.

Architecture
────────────
                ┌──────────────────────────────────────────┐
                │  For each timestep t = 0, 15, …, 180 min │
                │                                          │
  x_t (N,4)    │   ┌───────────────────┐                  │
  edge_index   │──▸│ InputProjection   │──▸ h_msg (N, H)  │
  edge_attr    │   │ (Linear + NNConv  │                  │
               │   │  × K layers)      │                  │
               │   └───────────────────┘                  │
               │             │                             │
               │             ▼                             │
               │   ┌───────────────────┐                  │
  h_{t-1}     │──▸│  GRUCell          │──▸ h_t (N, H)    │
               │   └───────────────────┘                  │
               │             │                             │
               │             ▼                             │
               │   ┌───────────────────┐                  │
               │   │  ReadoutMLP       │──▸ ŷ_t (N,)      │
               │   └───────────────────┘                  │
               └──────────────────────────────────────────┘

The model produces one surcharge-volume prediction per node per timestep,
matching the full 3-hr time-series requirement.

Usage (training):
    model = FloodGNN(n_node_feat=4, n_edge_feat=4)
    loss = model.training_step(sequence)  # list[Data]

Usage (inference):
    preds = model.predict(sequence)  # (T, N) tensor

References:
    NNConv — "Neural Message Passing for Quantum Chemistry" (Gilmer+ 2017)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv, BatchNorm


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MPNNLayer(nn.Module):
    """One round of edge-conditioned message passing (NNConv) + batch norm + ReLU.

    NNConv uses a small MLP (the "edge network") to compute a per-edge
    weight matrix from edge features.  This is ideal for our drainage
    network: pipe diameter, slope, length, and Manning's n directly
    condition how messages flow between junctions.
    """

    def __init__(self, in_channels: int, out_channels: int, n_edge_feat: int):
        super().__init__()
        # Edge network: maps edge features → (in_channels × out_channels) weight matrix
        self.edge_nn = nn.Sequential(
            nn.Linear(n_edge_feat, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels * out_channels),
        )
        self.conv = NNConv(
            in_channels, out_channels,
            nn=self.edge_nn,
            aggr="add",
        )
        self.bn = BatchNorm(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x, edge_index, edge_attr)))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FloodGNN(nn.Module):
    """Spatio-temporal GNN for per-node surcharge volume prediction.

    Targets are log-transformed (log1p) during training to handle the
    highly skewed surcharge distribution (0–22,800 m³).  Predictions
    are converted back via expm1 at inference time.

    Args:
        n_node_feat:  Number of input node features (default 4).
        n_edge_feat:  Number of edge features (default 4).
        hidden_dim:   Hidden dimension for message passing and GRU (default 64).
        n_mp_layers:  Number of message-passing rounds per timestep (default 3).
        dropout:      Dropout rate for the readout MLP (default 0.1).
        log_targets:  If True, train on log1p(y) and invert at inference.
    """

    def __init__(
        self,
        n_node_feat: int = 6,
        n_edge_feat: int = 4,
        hidden_dim: int = 64,
        n_mp_layers: int = 3,
        dropout: float = 0.1,
        log_targets: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mp_layers = n_mp_layers
        self.log_targets = log_targets

        # Input projection: node features → hidden_dim
        self.input_proj = nn.Linear(n_node_feat, hidden_dim)

        # Message-passing stack
        self.mp_layers = nn.ModuleList()
        for _ in range(n_mp_layers):
            self.mp_layers.append(MPNNLayer(hidden_dim, hidden_dim, n_edge_feat))

        # Temporal: GRU cell updates hidden state across timesteps
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Readout: hidden → surcharge volume (scalar per node)
        # Output is unbounded when log_targets=True (log-space),
        # Softplus when predicting raw volumes.
        readout_layers = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ]
        if not log_targets:
            readout_layers.append(nn.Softplus())
        self.readout = nn.Sequential(*readout_layers)

        # MSE in log-space: strong gradients on the large errors that matter most
        self.loss_fn = nn.MSELoss()

    def _encode_timestep(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Run input projection + K rounds of message passing for one timestep.

        Returns:
            h_msg: (N, hidden_dim) — spatial encoding of this timestep
        """
        h = F.relu(self.input_proj(x))
        for mp in self.mp_layers:
            h = mp(h, edge_index, edge_attr) + h  # residual connection
        return h

    def forward(self, sequence: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass over a temporal sequence of PyG Data objects.

        Args:
            sequence: list[Data] of length T.  Each Data has:
                - x:           (N, F_node)
                - edge_index:  (2, E)
                - edge_attr:   (E, F_edge)
                - y:           (N,) targets (only used in training_step)

        Returns:
            predictions: (T, N) predicted surcharge volumes
            targets:     (T, N) ground-truth surcharge volumes
        """
        T = len(sequence)
        N = sequence[0].x.size(0)
        device = sequence[0].x.device

        h = torch.zeros(N, self.hidden_dim, device=device)
        predictions = []
        targets = []

        for data in sequence:
            # Spatial encoding for this timestep
            h_msg = self._encode_timestep(data.x, data.edge_index, data.edge_attr)

            # Temporal update: GRU fuses spatial encoding with previous hidden state
            h = self.gru(h_msg, h)

            # Predict surcharge volume per node
            pred = self.readout(h).squeeze(-1)  # (N,)
            predictions.append(pred)

            # Log-transform targets to match prediction space
            y = torch.log1p(data.y) if self.log_targets else data.y
            targets.append(y)

        return torch.stack(predictions), torch.stack(targets)  # (T, N), (T, N)

    def training_step(self, sequence: list) -> torch.Tensor:
        """Run forward + compute loss.  Returns scalar loss tensor."""
        preds, targets = self.forward(sequence)
        return self.loss_fn(preds, targets)

    @torch.no_grad()
    def predict(self, sequence: list) -> torch.Tensor:
        """Inference: return (T, N) surcharge volume predictions in real units (m³)."""
        self.eval()
        preds, _ = self.forward(sequence)
        # Convert from log-space back to real volumes
        if self.log_targets:
            preds = torch.expm1(preds).clamp(min=0)
        return preds


# ---------------------------------------------------------------------------
# CLI: model summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = FloodGNN()
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FloodGNN")
    print(f"  Hidden dim:    {model.hidden_dim}")
    print(f"  MP layers:     {model.n_mp_layers}")
    print(f"  Total params:  {total_params:,}")
    print(f"  Trainable:     {trainable:,}")

    # Dummy forward pass
    from torch_geometric.data import Data
    N, E, T = 25, 24, 13
    dummy_seq = []
    for _ in range(T):
        dummy_seq.append(Data(
            x=torch.randn(N, 4),
            edge_index=torch.randint(0, N, (2, E)),
            edge_attr=torch.randn(E, 4),
            y=torch.rand(N),
        ))
    preds, targets = model(dummy_seq)
    print(f"\n  Input:  {T} timesteps × {N} nodes × 4 features")
    print(f"  Output: preds={list(preds.shape)}, targets={list(targets.shape)}")
    print(f"  Pred range: [{preds.min().item():.4f}, {preds.max().item():.4f}]")
