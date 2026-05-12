"""
evaluate.py — Évaluation complète du modèle sur le test set.

Métriques :
  - MAE, RMSE (régression)
  - Corrélation de Spearman
  - AUC-ROC (binarisation à 0.5)
  - Courbes loss train/val
  - Carte de risque moyen par cellule de grille
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HIGH_RISK_THRESH = 0.5

# Grille méditerranéenne
LAT_MIN, LAT_MAX = 35.0, 47.0
LON_MIN, LON_MAX = -5.0, 28.0
RES = 0.25
LATS = np.arange(LAT_MIN, LAT_MAX + RES, RES)
LONS = np.arange(LON_MIN, LON_MAX + RES, RES)
N_LAT, N_LON = len(LATS), len(LONS)


# ---------------------------------------------------------------------------
# Métriques de régression
# ---------------------------------------------------------------------------

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.abs(y_true - y_pred).mean())


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Corrélation de rang de Spearman via scipy.

    Mesure la monotonie de la relation prédiction/vérité terrain.
    """
    from scipy.stats import spearmanr
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = HIGH_RISK_THRESH) -> float:
    """
    AUC-ROC après binarisation des labels à `threshold`.

    Args:
        y_true     : scores de risque vrais ∈ [0, 1]
        y_pred     : scores prédits ∈ [0, 1]
        threshold  : seuil de binarisation
    Returns:
        AUC-ROC ∈ [0, 1]
    """
    from sklearn.metrics import roc_auc_score
    y_bin = (y_true >= threshold).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        logger.warning("AUC indéfinie (toutes les étiquettes identiques) → 0.5")
        return 0.5
    return float(roc_auc_score(y_bin, y_pred))


# ---------------------------------------------------------------------------
# Inférence sur un DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lance l'inférence sur tous les batches d'un loader.

    Returns:
        (y_true, y_pred) arrays 1D de même longueur
    """
    model.eval()
    all_pred, all_true = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch).cpu().numpy()
        all_pred.append(pred)
        all_true.append(y_batch.numpy())

    return np.concatenate(all_true), np.concatenate(all_pred)


# ---------------------------------------------------------------------------
# Évaluation complète
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    output_dir: Path,
    history_path: Optional[Path] = None,
    model_name: str = "lstm",
    device: Optional[torch.device] = None,
    n_lat: int = N_LAT,
    n_lon: int = N_LON,
) -> dict[str, float]:
    """
    Évalue le modèle sur le test set et produit toutes les visualisations.

    Args:
        model        : modèle chargé (état best_model.pt)
        test_loader  : DataLoader test
        output_dir   : répertoire de sortie pour les figures et métriques
        history_path : chemin vers history_{model}.json (pour les courbes)
        model_name   : nom du modèle pour les titres/fichiers
        device       : device PyTorch
        n_lat, n_lon : dimensions de la grille pour la carte de risque
    Returns:
        dict des métriques scalaires
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    logger.info(f"Inférence sur le test set ({model_name}) …")
    y_true, y_pred = predict(model, test_loader, device)

    mae     = compute_mae(y_true, y_pred)
    rmse    = compute_rmse(y_true, y_pred)
    spear   = compute_spearman(y_true, y_pred)
    auc     = compute_auc(y_true, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "Spearman": spear, "AUC": auc}
    logger.info(f"MAE={mae:.4f}  RMSE={rmse:.4f}  Spearman={spear:.4f}  AUC={auc:.4f}")

    # Sauvegarde JSON
    metrics_path = output_dir / f"metrics_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Métriques → {metrics_path}")

    # Courbes de loss
    if history_path and history_path.exists():
        _plot_loss_curves(history_path, output_dir, model_name)

    # Carte de risque moyen
    _plot_risk_map(y_pred, output_dir, model_name, n_lat, n_lon)

    # Scatter plot : vrai vs prédit
    _plot_scatter(y_true, y_pred, output_dir, model_name, metrics)

    return metrics


# ---------------------------------------------------------------------------
# Visualisations internes
# ---------------------------------------------------------------------------

def _plot_loss_curves(history_path: Path, output_dir: Path, model_name: str):
    """Trace et sauvegarde les courbes train/val loss."""
    import matplotlib.pyplot as plt

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train", linewidth=2)
    ax.plot(history["val_loss"],   label="Val",   linewidth=2)
    ax.set_xlabel("Époque")
    ax.set_ylabel("Weighted MSE Loss")
    ax.set_title(f"Courbes de loss — {model_name.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE val
    ax = axes[1]
    ax.plot(history["val_mae"], color="darkorange", linewidth=2, label="Val MAE")
    ax.set_xlabel("Époque")
    ax.set_ylabel("MAE")
    ax.set_title(f"MAE de validation — {model_name.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / f"loss_curves_{model_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Courbes de loss → {out_path}")


def _plot_risk_map(
    y_pred: np.ndarray,
    output_dir: Path,
    model_name: str,
    n_lat: int,
    n_lon: int,
):
    """
    Carte de risque moyen par cellule de grille.

    Si le nombre de prédictions est un multiple de n_lat*n_lon, regroupe
    par cellule. Sinon, distribue aléatoirement sur la grille.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_cells = n_lat * n_lon

    if len(y_pred) >= n_cells and len(y_pred) % n_cells == 0:
        risk_map = y_pred.reshape(-1, n_lat, n_lon).mean(axis=0)
    else:
        # Distribution sur la grille en pliant les prédictions disponibles
        padded = np.resize(y_pred, n_cells)
        risk_map = padded.reshape(n_lat, n_lon)

    lats = np.arange(LAT_MIN, LAT_MAX + RES, RES)
    lons = np.arange(LON_MIN, LON_MAX + RES, RES)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fire_risk", ["#2b83ba", "#ffffbf", "#d7191c"]
    )
    im = ax.pcolormesh(lons, lats, risk_map, cmap=cmap, vmin=0, vmax=1, shading="auto")
    plt.colorbar(im, ax=ax, label="Risque moyen prédit ∈ [0, 1]", shrink=0.8)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Carte de risque incendie — {model_name.upper()} (Méditerranée)")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.grid(True, alpha=0.3, linestyle="--")

    out_path = output_dir / f"risk_map_{model_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Carte de risque → {out_path}")


def _plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    model_name: str,
    metrics: dict[str, float],
):
    """Scatter plot vrai vs prédit avec métriques annotées."""
    import matplotlib.pyplot as plt

    # Sous-échantillonnage pour la lisibilité (max 5000 points)
    idx = np.random.choice(len(y_true), size=min(5000, len(y_true)), replace=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true[idx], y_pred[idx], alpha=0.3, s=5, color="#e63946")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Parfait")
    ax.set_xlabel("Risk score réel")
    ax.set_ylabel("Risk score prédit")
    ax.set_title(f"Réel vs Prédit — {model_name.upper()}")
    ax.text(0.05, 0.92,
            f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}\n"
            f"Spearman={metrics['Spearman']:.4f}  AUC={metrics['AUC']:.4f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    out_path = output_dir / f"scatter_{model_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Scatter → {out_path}")


# ---------------------------------------------------------------------------
# Chargement du modèle depuis checkpoint
# ---------------------------------------------------------------------------

def load_best_model(checkpoint_path: Path, model: nn.Module) -> nn.Module:
    """
    Charge les poids du meilleur checkpoint dans le modèle.

    Args:
        checkpoint_path : chemin vers best_model_{name}.pt
        model           : instance du modèle avec la bonne architecture
    Returns:
        modèle avec poids chargés, en mode eval
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    logger.info(f"Modèle chargé depuis epoch {epoch}  val_loss={val_loss:.5f}")
    return model


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """Évalue le meilleur LSTM entraîné (mode simulé si pas de données réelles)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from dataset import build_dataloaders
    from model import build_model
    from train import set_seed

    set_seed(42)

    base          = Path(__file__).resolve().parents[1]
    processed_dir = base / "data" / "processed"
    figures_dir   = base / "data" / "figures"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, _ = build_dataloaders(
        processed_dir, batch_size=512, simulate=True
    )

    for model_name in ("lstm", "tcn"):
        ckpt_path = processed_dir / f"best_model_{model_name}.pt"
        model = build_model(model_name, n_features=8)

        if ckpt_path.exists():
            model = load_best_model(ckpt_path, model)
        else:
            logger.warning(f"Checkpoint {ckpt_path} absent → évaluation sans entraînement.")

        history_path = processed_dir / f"history_{model_name}.json"
        metrics = evaluate_model(
            model, test_loader,
            output_dir=figures_dir,
            history_path=history_path,
            model_name=model_name,
            device=device,
        )
        print(f"\n=== {model_name.upper()} ===")
        for k, v in metrics.items():
            print(f"  {k:12s} : {v:.4f}")


if __name__ == "__main__":
    main()
