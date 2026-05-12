"""
train.py — Boucle d'entraînement complète avec early stopping et logging.

Loss     : Weighted MSE (poids=10 pour risk > 0.5)
Optimizer: Adam lr=0.001
Scheduler: ReduceLROnPlateau
Early stopping : patience=10 epochs
Sauvegarde    : best_model.pt + scaler.pkl
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HIGH_RISK_WEIGHT = 10.0   # poids pour les cellules à risque > 0.5
HIGH_RISK_THRESH = 0.5


# ---------------------------------------------------------------------------
# Fonctions de perte
# ---------------------------------------------------------------------------

def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE pondérée : les échantillons à risque élevé (> HIGH_RISK_THRESH)
    reçoivent un poids de HIGH_RISK_WEIGHT pour compenser le déséquilibre.

    Args:
        pred   : prédictions (batch,) ∈ [0, 1]
        target : labels vrais (batch,) ∈ [0, 1]
    Returns:
        loss scalaire
    """
    weights = torch.where(target > HIGH_RISK_THRESH,
                          torch.full_like(target, HIGH_RISK_WEIGHT),
                          torch.ones_like(target))
    squared_errors = (pred - target) ** 2
    return (weights * squared_errors).mean()


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Fixe tous les seeds pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Boucle d'une époque
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Lance une époque d'entraînement.

    Args:
        model     : modèle PyTorch
        loader    : DataLoader train
        optimizer : optimiseur
        device    : cpu ou cuda
    Returns:
        loss moyenne sur l'époque
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = weighted_mse_loss(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Évalue le modèle sur un loader (val ou test).

    Returns:
        (weighted_mse_loss, mae)
    """
    model.eval()
    total_wmse = 0.0
    total_mae  = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(X_batch)
        wmse = weighted_mse_loss(pred, y_batch)
        mae  = (pred - y_batch).abs().mean()

        total_wmse += wmse.item() * len(y_batch)
        total_mae  += mae.item()  * len(y_batch)

    n = len(loader.dataset)
    return total_wmse / n, total_mae / n


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Arrête l'entraînement si la val loss ne s'améliore pas pendant `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        """Retourne True si l'entraînement doit s'arrêter."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Boucle principale d'entraînement
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    n_epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    device: Optional[torch.device] = None,
    model_name: str = "lstm",
) -> dict[str, list[float]]:
    """
    Boucle d'entraînement complète.

    Args:
        model        : modèle PyTorch
        train_loader : DataLoader train
        val_loader   : DataLoader val
        output_dir   : où sauvegarder best_model.pt et history.json
        n_epochs     : nombre max d'époques
        lr           : learning rate initial
        patience     : patience early stopping
        device       : device PyTorch (auto-détecté si None)
        model_name   : préfixe pour le fichier de sauvegarde
    Returns:
        history dict avec listes train_loss, val_loss, val_mae
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stop = EarlyStopping(patience=patience)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path  = output_dir / f"best_model_{model_name}.pt"

    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae = evaluate_epoch(model, val_loader, device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:03d}/{n_epochs}  "
            f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"val_mae={val_mae:.4f}  lr={lr_now:.6f}  {dt:.1f}s"
        )

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":   val_loss,
                "val_mae":    val_mae,
            }, best_path)
            logger.info(f"  ✓ Meilleur modèle sauvegardé (val_loss={val_loss:.5f})")

        scheduler.step(val_loss)

        if early_stop.step(val_loss):
            logger.info(f"Early stopping après {epoch} époques.")
            break

    # Sauvegarde de l'historique
    history_path = output_dir / f"history_{model_name}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Historique sauvegardé : {history_path}")

    return history


# ---------------------------------------------------------------------------
# main — démonstration complète (mode simulé)
# ---------------------------------------------------------------------------

def main():
    """Entraîne un LSTM et un TCN en mode simulation et affiche les courbes."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from dataset import build_dataloaders
    from model import build_model

    set_seed(42)

    base = Path(__file__).resolve().parents[1]
    processed_dir = base / "data" / "processed"
    output_dir    = base / "data" / "processed"

    logger.info("=== ENTRAÎNEMENT (mode simulé) ===")

    train_loader, val_loader, test_loader, scaler = build_dataloaders(
        processed_dir, batch_size=512, simulate=True
    )

    # Sauvegarde du scaler
    scaler.save(processed_dir / "scaler.pkl")

    for model_type in ("lstm", "tcn"):
        logger.info(f"\n--- Modèle : {model_type.upper()} ---")
        model = build_model(model_type, n_features=8)
        history = train(
            model, train_loader, val_loader,
            output_dir=output_dir,
            n_epochs=30,
            patience=5,
            model_name=model_type,
        )
        logger.info(
            f"Fin entraînement {model_type}  "
            f"| best val_loss = {min(history['val_loss']):.5f}"
        )


if __name__ == "__main__":
    main()
