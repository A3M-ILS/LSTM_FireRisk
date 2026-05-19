"""
train.py — Entraînement LSTM + TCN avec split temporel strict.

Charge X_train.npy, y_train.npy, X_val.npy, y_val.npy depuis data/processed/.
Weighted MSE Loss (poids=10 si risk > 0.5).
Early stopping patience=10 sur val_loss.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HIGH_RISK_WEIGHT = 10.0
HIGH_RISK_THRESH = 0.5
SEQ_LEN = 15
N_FEATURES = 8


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Construit des fenêtres glissantes (seq_len, n_features) depuis
    un array (T, N_LAT, N_LON, F).
    """

    def __init__(self, X_path: Path, y_path: Path, seq_len: int = SEQ_LEN):
        X_raw = np.load(X_path)   # (T, N_LAT, N_LON, F)
        y_raw = np.load(y_path)   # (T, N_LAT, N_LON)

        T, NL, NG, F = X_raw.shape
        N_cells = NL * NG

        X_flat = X_raw.reshape(T, N_cells, F)     # (T, N_cells, F)
        y_flat = y_raw.reshape(T, N_cells)         # (T, N_cells)

        n_win = T - seq_len + 1
        seqs_X = np.zeros((N_cells, n_win, seq_len, F), dtype=np.float32)
        seqs_y = np.zeros((N_cells, n_win), dtype=np.float32)

        for t in range(n_win):
            seqs_X[:, t, :, :] = X_flat[t:t + seq_len, :, :].transpose(1, 0, 2)
            seqs_y[:, t]        = y_flat[t + seq_len - 1, :]

        self.X = seqs_X.reshape(-1, seq_len, F)
        self.y = seqs_y.reshape(-1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)

    def sample_weights(self) -> np.ndarray:
        w = np.ones(len(self.y), dtype=np.float32)
        w[self.y > HIGH_RISK_THRESH] = HIGH_RISK_WEIGHT
        return w


# ---------------------------------------------------------------------------
# Loss pondérée
# ---------------------------------------------------------------------------

def weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    w = torch.where(target > HIGH_RISK_THRESH,
                    torch.full_like(target, HIGH_RISK_WEIGHT),
                    torch.ones_like(target))
    return (w * (pred - target) ** 2).mean()


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Boucle d'entraînement d'une époque
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = total_mae = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            if train:
                optimizer.zero_grad()
            pred = model(X_b)
            loss = weighted_mse(pred, y_b)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y_b)
            total_mae  += (pred.detach() - y_b).abs().mean().item() * len(y_b)

    n = len(loader.dataset)
    return total_loss / n, total_mae / n


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    model_name: str = "lstm",
    n_epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    device: Optional[torch.device] = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    stopper   = EarlyStopping(patience=patience)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"best_model_{model_name}.pt"

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    best_val_loss = float("inf")

    logger.info(f"\n{'='*60}")
    logger.info(f"  Modèle : {model_name.upper()} | Device : {device} | Epochs max : {n_epochs}")
    logger.info(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        tr_loss, tr_mae = run_epoch(model, train_loader, optimizer, device, train=True)
        vl_loss, vl_mae = run_epoch(model, val_loader,   optimizer, device, train=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(vl_mae)

        scheduler.step(vl_loss)
        dt = time.time() - t0

        saved = ""
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": vl_loss, "val_mae": vl_mae}, best_path)
            saved = " [best]"

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f} | "
            f"MAE: {vl_mae:.4f} | lr={lr_now:.2e} | {dt:.1f}s{saved}"
        )

        if stopper.step(vl_loss):
            logger.info(f"Early stopping à l'époque {epoch} (patience={patience})")
            break

    hist_path = output_dir / f"history_{model_name}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Historique -> {hist_path}")
    logger.info(f"Meilleure val_loss : {best_val_loss:.5f}")

    return history


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model

    set_seed(42)

    base          = Path(__file__).resolve().parents[1]
    processed_dir = base / "data" / "processed"

    # Vérifier si les .npy sont présents
    if not (processed_dir / "X_train.npy").exists():
        logger.error("X_train.npy absent. Lance d'abord : python src/pipeline.py")
        sys.exit(1)

    logger.info("Chargement des datasets...")
    train_ds = SequenceDataset(
        processed_dir / "X_train.npy",
        processed_dir / "y_train.npy",
    )
    val_ds = SequenceDataset(
        processed_dir / "X_val.npy",
        processed_dir / "y_val.npy",
    )

    logger.info(f"Train : {len(train_ds):,} séquences | Val : {len(val_ds):,} séquences")
    logger.info(f"Risque élevé (>0.5) dans train : {(train_ds.y > 0.5).mean()*100:.1f}%")

    # Weighted sampler
    weights = train_ds.sample_weights()
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(weights), replacement=True
    )

    BATCH = 512
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")

    for model_type in ("lstm", "tcn"):
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Entraînement {model_type.upper()}")
        logger.info(f"{'─'*60}")

        model = build_model(model_type, n_features=N_FEATURES)
        history = train_model(
            model, train_loader, val_loader,
            output_dir=processed_dir,
            model_name=model_type,
            n_epochs=100,
            patience=10,
            device=device,
        )

        best_v = min(history["val_loss"])
        best_m = min(history["val_mae"])
        logger.info(
            f"\n{model_type.upper()} termine -> "
            f"best val_loss={best_v:.5f} | best val_mae={best_m:.5f} "
            f"| {len(history['train_loss'])} époques"
        )

    logger.info("\nEntrainement complet. Modeles dans data/processed/")


if __name__ == "__main__":
    main()
