"""
dataset.py — PyTorch Dataset pour séquences LSTM de prédiction de risque incendie.

Génère des fenêtres glissantes de 15 jours (SEQ_LEN) sur la grille 0.25°.
Normalisation MinMaxScaler fittée sur le train uniquement.
Weighted sampler pour gérer le fort déséquilibre risque élevé/faible.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

FEATURE_NAMES = [
    "ndvi",
    "temp_max",
    "humidity",
    "wind_speed",
    "precip",
    "drought_index",
    "past_hotspot_count",
    "past_frp",
]
N_FEATURES = len(FEATURE_NAMES)
SEQ_LEN = 15          # jours de contexte
HIGH_RISK_THRESH = 0.5  # seuil pour le weighted sampler


# ---------------------------------------------------------------------------
# MinMaxScaler adapté aux arrays 3D+ (T, …, F)
# ---------------------------------------------------------------------------

class FeatureScaler:
    """
    MinMax normalisation par feature sur l'axe temporal.

    S'adapte sur les données d'entraînement et transforme les autres splits.
    """

    def __init__(self):
        self.min_: Optional[np.ndarray] = None   # shape (N_FEATURES,)
        self.max_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """
        Calcule min/max par feature.

        Args:
            X : array de shape (T, N_LAT, N_LON, N_FEATURES) ou (N, N_FEATURES)
        """
        flat = X.reshape(-1, X.shape[-1])
        self.min_ = flat.min(axis=0)
        self.max_ = flat.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalise X dans [0, 1] par feature."""
        if self.min_ is None:
            raise RuntimeError("Appeler .fit() avant .transform()")
        rng = self.max_ - self.min_
        rng[rng == 0] = 1.0   # évite division par zéro
        return ((X - self.min_) / rng).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"min_": self.min_, "max_": self.max_}, f)

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        scaler = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        scaler.min_ = data["min_"]
        scaler.max_ = data["max_"]
        return scaler


# ---------------------------------------------------------------------------
# Dataset principal
# ---------------------------------------------------------------------------

class FireRiskDataset(Dataset):
    """
    Dataset PyTorch qui produit des fenêtres (séquences) de SEQ_LEN jours.

    Chaque échantillon :
        X : (SEQ_LEN, N_FEATURES)   features normalisées d'une cellule
        y : scalar float32          risk_score du jour cible (t + SEQ_LEN)

    Args:
        X_path   : chemin vers X_{split}.npy, shape (T, N_LAT, N_LON, N_FEATURES)
        y_path   : chemin vers y_{split}.npy, shape (T, N_LAT, N_LON)
        scaler   : FeatureScaler déjà fitté (None si split=train → fit automatique)
        seq_len  : longueur de séquence
        is_train : si True, fitte le scaler sur ces données
    """

    def __init__(
        self,
        X_path: Path,
        y_path: Path,
        scaler: Optional[FeatureScaler] = None,
        seq_len: int = SEQ_LEN,
        is_train: bool = False,
    ):
        X_raw = np.load(X_path)   # (T, N_LAT, N_LON, F)
        y_raw = np.load(y_path)   # (T, N_LAT, N_LON)

        T, N_LAT, N_LON, F = X_raw.shape

        # Normalisation
        if is_train:
            self.scaler = FeatureScaler()
            X_norm = self.scaler.fit_transform(X_raw)
        else:
            if scaler is None:
                raise ValueError("Fournir un scaler fitté pour val/test")
            self.scaler = scaler
            X_norm = scaler.transform(X_raw)

        # Aplatir les cellules spatiales : (T, N_cells, F)
        N_cells = N_LAT * N_LON
        X_flat = X_norm.reshape(T, N_cells, F)   # (T, N_cells, F)
        y_flat = y_raw.reshape(T, N_cells)         # (T, N_cells)

        # Construire les séquences glissantes
        # Pour chaque cellule k et chaque t ∈ [0, T-seq_len) :
        #   X[k, t] = X_flat[t : t+seq_len, k, :]   (seq_len, F)
        #   y[k, t] = y_flat[t + seq_len - 1, k]    (scalar)
        n_windows = T - seq_len + 1
        sequences_X = np.zeros((N_cells, n_windows, seq_len, F), dtype=np.float32)
        sequences_y = np.zeros((N_cells, n_windows), dtype=np.float32)

        for t in range(n_windows):
            sequences_X[:, t, :, :] = X_flat[t:t + seq_len, :, :].transpose(1, 0, 2)
            sequences_y[:, t] = y_flat[t + seq_len - 1, :]

        # Mise à plat finale : (N_cells * n_windows, seq_len, F)
        self.X = sequences_X.reshape(-1, seq_len, F)
        self.y = sequences_y.reshape(-1)

        self.seq_len = seq_len
        self.n_features = F

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])           # (seq_len, F)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

    def get_sample_weights(self) -> np.ndarray:
        """
        Calcule les poids par échantillon pour le WeightedRandomSampler.

        Les cellules à risque élevé (y > HIGH_RISK_THRESH) reçoivent un poids 10×.
        """
        weights = np.ones(len(self.y), dtype=np.float32)
        high_risk_mask = self.y > HIGH_RISK_THRESH
        weights[high_risk_mask] = 10.0
        return weights


# ---------------------------------------------------------------------------
# Simulation de données (test sans pipeline)
# ---------------------------------------------------------------------------

def make_simulated_arrays(
    n_days: int = 200,
    n_lat: int = 10,
    n_lon: int = 10,
    n_features: int = N_FEATURES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère des arrays (T, N_LAT, N_LON, F) et (T, N_LAT, N_LON) simulés.

    Utile pour les tests unitaires et la démo sans données réelles.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_days, n_lat, n_lon, n_features)).astype(np.float32)
    # Risque corrélé à la température (feature index 1) et NDVI inversé
    risk = (X[..., 1] * 0.4 + (1 - X[..., 0]) * 0.3 + rng.random((n_days, n_lat, n_lon)) * 0.3)
    y = np.clip(risk, 0.0, 1.0).astype(np.float32)
    return X, y


class SimulatedFireRiskDataset(Dataset):
    """
    Dataset 100% simulé, sans dépendance aux fichiers .npy.
    Utile pour tester le modèle et la boucle d'entraînement.
    """

    def __init__(
        self,
        n_days: int = 200,
        n_lat: int = 10,
        n_lon: int = 10,
        seq_len: int = SEQ_LEN,
        seed: int = 42,
        scaler: Optional[FeatureScaler] = None,
        is_train: bool = False,
    ):
        X_raw, y_raw = make_simulated_arrays(n_days, n_lat, n_lon, seed=seed)
        T, N_LAT, N_LON, F = X_raw.shape

        if is_train:
            self.scaler = FeatureScaler()
            X_norm = self.scaler.fit_transform(X_raw)
        else:
            if scaler is None:
                self.scaler = FeatureScaler()
                X_norm = self.scaler.fit_transform(X_raw)
            else:
                self.scaler = scaler
                X_norm = scaler.transform(X_raw)

        N_cells = N_LAT * N_LON
        X_flat = X_norm.reshape(T, N_cells, F)
        y_flat = y_raw.reshape(T, N_cells)

        n_windows = T - seq_len + 1
        sequences_X = np.zeros((N_cells, n_windows, seq_len, F), dtype=np.float32)
        sequences_y = np.zeros((N_cells, n_windows), dtype=np.float32)

        for t in range(n_windows):
            sequences_X[:, t, :, :] = X_flat[t:t + seq_len, :, :].transpose(1, 0, 2)
            sequences_y[:, t] = y_flat[t + seq_len - 1, :]

        self.X = sequences_X.reshape(-1, seq_len, F)
        self.y = sequences_y.reshape(-1)
        self.seq_len = seq_len
        self.n_features = F

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

    def get_sample_weights(self) -> np.ndarray:
        weights = np.ones(len(self.y), dtype=np.float32)
        weights[self.y > HIGH_RISK_THRESH] = 10.0
        return weights


# ---------------------------------------------------------------------------
# Factory de DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    processed_dir: Path,
    batch_size: int = 256,
    seq_len: int = SEQ_LEN,
    num_workers: int = 0,
    simulate: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, FeatureScaler]:
    """
    Construit les trois DataLoaders (train / val / test).

    Si simulate=True ou si les fichiers .npy sont absents, utilise
    SimulatedFireRiskDataset.

    Args:
        processed_dir : répertoire data/processed/
        batch_size    : taille de batch
        seq_len       : longueur de séquence LSTM
        num_workers   : workers DataLoader (0 sur Windows)
        simulate      : forcer la simulation
    Returns:
        (train_loader, val_loader, test_loader, scaler)
    """
    x_train = processed_dir / "X_train.npy"
    y_train = processed_dir / "y_train.npy"

    if simulate or not x_train.exists():
        train_ds = SimulatedFireRiskDataset(n_days=400, seq_len=seq_len, seed=0, is_train=True)
        val_ds   = SimulatedFireRiskDataset(n_days=150, seq_len=seq_len, seed=1, scaler=train_ds.scaler)
        test_ds  = SimulatedFireRiskDataset(n_days=100, seq_len=seq_len, seed=2, scaler=train_ds.scaler)
    else:
        train_ds = FireRiskDataset(x_train, y_train, seq_len=seq_len, is_train=True)
        scaler   = train_ds.scaler
        val_ds   = FireRiskDataset(
            processed_dir / "X_val.npy", processed_dir / "y_val.npy",
            scaler=scaler, seq_len=seq_len,
        )
        test_ds  = FireRiskDataset(
            processed_dir / "X_test.npy", processed_dir / "y_test.npy",
            scaler=scaler, seq_len=seq_len,
        )

    # Weighted sampler pour le train
    weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_ds.scaler


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """Teste le Dataset en mode simulation."""
    import time

    print("=== TEST DATASET (mode simulé) ===")
    t0 = time.time()

    train_ds = SimulatedFireRiskDataset(n_days=300, is_train=True, seed=0)
    val_ds   = SimulatedFireRiskDataset(n_days=100, scaler=train_ds.scaler, seed=1)

    print(f"Train : {len(train_ds)} séquences")
    print(f"Val   : {len(val_ds)} séquences")

    x, y = train_ds[0]
    print(f"x.shape = {x.shape}  (seq_len={SEQ_LEN}, n_features={N_FEATURES})")
    print(f"y       = {y.item():.4f}")
    print(f"Risque élevé (>0.5) dans train : {(train_ds.y > 0.5).mean() * 100:.1f}%")
    print(f"Durée : {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
