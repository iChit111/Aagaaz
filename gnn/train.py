"""Training loop for FloodGNN.

Loads the training dataset from .npz files, splits into train/val/test,
trains the GNN+GRU model with:
    - Adam optimiser with cosine-annealing LR schedule
    - Early stopping on validation loss
    - Model checkpointing (best val loss)
    - TensorBoard logging
    - GPU support (auto-detected)

Usage:
    # Quick test (2 epochs, CPU)
    python3 -m gnn.train --max-epochs 2 --device cpu

    # Full training on GPU
    python3 -m gnn.train --data-dir data/gnn_training --max-epochs 200

    # Resume from checkpoint
    python3 -m gnn.train --resume checkpoints/flood_gnn_best.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from gnn.graph_builder import TrainingDataset, train_val_test_split
from gnn.model import FloodGNN

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Compute regression metrics on (T, N) tensors.

    Returns dict with keys: mse, rmse, mae, r2, peak_error_pct.
    """
    mse = nn.functional.mse_loss(preds, targets).item()
    rmse = math.sqrt(mse)
    mae = (preds - targets).abs().mean().item()

    # R² (coefficient of determination)
    ss_res = ((targets - preds) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1 - (ss_res / max(ss_tot, 1e-8))

    # Peak error: how well we predict the worst-case surcharge
    # This matters most for flood nowcasting — getting the peak wrong is dangerous
    peak_pred = preds.max().item()
    peak_true = targets.max().item()
    peak_error_pct = (
        abs(peak_pred - peak_true) / max(peak_true, 1e-8) * 100
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "peak_error_pct": peak_error_pct,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: str | Path = "data/gnn_training",
    checkpoint_dir: str | Path = "checkpoints",
    max_epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim: int = 128,
    n_mp_layers: int = 3,
    dropout: float = 0.1,
    device: str | None = None,
    resume: str | None = None,
    log_dir: str | Path = "runs/flood_gnn",
    grad_clip: float = 1.0,
    seed: int = 42,
) -> Path:
    """Train FloodGNN and return path to the best checkpoint.

    Args:
        data_dir:       Directory containing .npz scenario files.
        checkpoint_dir: Where to save model checkpoints.
        max_epochs:     Maximum training epochs.
        patience:       Early-stopping patience (epochs without val improvement).
        lr:             Initial learning rate.
        weight_decay:   AdamW weight decay.
        hidden_dim:     GNN hidden dimension.
        n_mp_layers:    Number of message-passing layers.
        dropout:        Dropout rate in readout MLP.
        device:         'cuda', 'cpu', or None (auto-detect).
        resume:         Path to checkpoint to resume from.
        log_dir:        TensorBoard log directory.
        grad_clip:      Max gradient norm for clipping.
        seed:           Random seed.

    Returns:
        Path to the best checkpoint file.
    """
    torch.manual_seed(seed)

    # Device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Load dataset
    logger.info("Loading dataset from %s", data_dir)
    dataset = TrainingDataset(data_dir)
    logger.info("Dataset: %d scenarios", len(dataset))

    # Split
    train_idx, val_idx, test_idx = train_val_test_split(dataset, seed=seed)
    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )

    # Model
    model = FloodGNN(
        n_node_feat=dataset.n_node_features,
        n_edge_feat=dataset.n_edge_features,
        hidden_dim=hidden_dim,
        n_mp_layers=n_mp_layers,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s params", f"{total_params:,}")

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info("Resumed from epoch %d (best val loss: %.6f)", start_epoch, best_val_loss)

    # TensorBoard (optional)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info("TensorBoard logging to %s", log_dir)
    except ImportError:
        logger.info("TensorBoard not available — skipping logging")

    # Checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "flood_gnn_best.pt"

    # Helper: move a sequence of Data objects to device
    def to_device(seq: list) -> list:
        return [data.to(device) for data in seq]

    # Helper: evaluate on a set of indices
    @torch.no_grad()
    def evaluate(indices: list[int]) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        all_preds_real = []
        all_targets_real = []
        for idx in indices:
            seq = to_device(dataset[idx])
            preds, targets = model(seq)
            total_loss += model.loss_fn(preds, targets).item()
            # Convert to real-space for metrics
            if model.log_targets:
                all_preds_real.append(torch.expm1(preds).clamp(min=0).cpu())
                all_targets_real.append(torch.expm1(targets).clamp(min=0).cpu())
            else:
                all_preds_real.append(preds.cpu())
                all_targets_real.append(targets.cpu())
        avg_loss = total_loss / max(len(indices), 1)
        # Aggregate metrics across all scenarios (in real-space m³)
        all_preds_real = torch.cat(all_preds_real, dim=0)
        all_targets_real = torch.cat(all_targets_real, dim=0)
        metrics = compute_metrics(all_preds_real, all_targets_real)
        metrics["loss"] = avg_loss
        return metrics

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    epochs_without_improvement = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        # Shuffle training order each epoch
        rng = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(len(train_idx), generator=rng).tolist()
        shuffled_train = [train_idx[i] for i in perm]

        for i, idx in enumerate(shuffled_train):
            seq = to_device(dataset[idx])

            optimizer.zero_grad()
            loss = model.training_step(seq)
            loss.backward()

            # Gradient clipping to stabilise training
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        avg_train_loss = epoch_loss / max(len(train_idx), 1)
        elapsed = time.time() - t0

        # Validation
        val_metrics = evaluate(val_idx)
        val_loss = val_metrics["loss"]

        # Logging
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  "
            "val_rmse=%.4f  val_r2=%.4f  peak_err=%.1f%%  "
            "lr=%.2e  time=%.1fs",
            epoch + 1, max_epochs,
            avg_train_loss, val_loss,
            val_metrics["rmse"], val_metrics["r2"],
            val_metrics["peak_error_pct"],
            current_lr, elapsed,
        )

        if writer:
            writer.add_scalar("loss/train", avg_train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("metrics/val_rmse", val_metrics["rmse"], epoch)
            writer.add_scalar("metrics/val_r2", val_metrics["r2"], epoch)
            writer.add_scalar("metrics/val_mae", val_metrics["mae"], epoch)
            writer.add_scalar("metrics/peak_error_pct", val_metrics["peak_error_pct"], epoch)
            writer.add_scalar("lr", current_lr, epoch)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "hidden_dim": hidden_dim,
                "n_mp_layers": n_mp_layers,
                "n_node_feat": dataset.n_node_features,
                "n_edge_feat": dataset.n_edge_features,
                "dropout": dropout,
                "log_targets": model.log_targets,
            }, best_ckpt_path)
            logger.info("  ✓ Saved best checkpoint (val_loss=%.6f)", best_val_loss)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch + 1, patience,
                )
                break

    # -----------------------------------------------------------------------
    # Final evaluation on test set
    # -----------------------------------------------------------------------

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded best checkpoint from epoch %d", ckpt["epoch"] + 1)

    test_metrics = evaluate(test_idx)
    logger.info("─" * 60)
    logger.info("TEST RESULTS (best checkpoint)")
    logger.info("  MSE:            %.6f", test_metrics["mse"])
    logger.info("  RMSE:           %.4f m³", test_metrics["rmse"])
    logger.info("  MAE:            %.4f m³", test_metrics["mae"])
    logger.info("  R²:             %.4f", test_metrics["r2"])
    logger.info("  Peak error:     %.1f%%", test_metrics["peak_error_pct"])
    logger.info("─" * 60)
    logger.info("Best checkpoint saved to: %s", best_ckpt_path)

    if writer:
        writer.add_hparams(
            {
                "hidden_dim": hidden_dim,
                "n_mp_layers": n_mp_layers,
                "lr": lr,
                "dropout": dropout,
                "weight_decay": weight_decay,
            },
            {
                "hparam/test_rmse": test_metrics["rmse"],
                "hparam/test_r2": test_metrics["r2"],
                "hparam/test_peak_err": test_metrics["peak_error_pct"],
            },
        )
        writer.close()

    return best_ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train FloodGNN for drainage surcharge prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data/gnn_training",
                        help="Directory containing .npz training scenarios")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="AdamW weight decay")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension for GNN and GRU")
    parser.add_argument("--n-mp-layers", type=int, default=3,
                        help="Number of message-passing layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate in readout MLP")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda', 'cpu', or auto-detect")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--log-dir", type=str, default="runs/flood_gnn",
                        help="TensorBoard log directory")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        n_mp_layers=args.n_mp_layers,
        dropout=args.dropout,
        device=args.device,
        resume=args.resume,
        log_dir=args.log_dir,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
