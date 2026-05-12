"""
model.py — Architectures de modèles pour la prédiction de risque incendie.

Modèles disponibles :
  - FireRiskLSTM     : LSTM 2 couches + tête dense + Sigmoid
  - FireRiskTCN      : Temporal Convolutional Network (baseline Conv1d)

Les deux produisent une sortie scalaire ∈ [0, 1] par cellule.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LSTM principal
# ---------------------------------------------------------------------------

class FireRiskLSTM(nn.Module):
    """
    LSTM bi-couche suivi d'une tête de régression avec Sigmoid.

    Architecture :
        LSTM(input=n_features, hidden=hidden_size, num_layers=2, dropout=dropout)
        → dernier état caché
        → Linear(hidden_size, 64) → ReLU → Dropout
        → Linear(64, 1) → Sigmoid

    Args:
        n_features  : nombre de features en entrée (8 par défaut)
        hidden_size : taille de l'état caché LSTM
        num_layers  : nombre de couches LSTM
        dropout     : dropout entre couches LSTM et dans la tête dense
    """

    def __init__(
        self,
        n_features: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, n_features)
        Returns:
            out : (batch,) valeurs ∈ [0, 1]
        """
        # lstm_out : (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(x)
        # On prend le dernier pas de temps
        last_hidden = lstm_out[:, -1, :]   # (batch, hidden)
        out = self.head(last_hidden)       # (batch, 1)
        return out.squeeze(-1)             # (batch,)

    def init_weights(self):
        """Initialisation orthogonale des poids LSTM, Xavier pour les couches denses."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


# ---------------------------------------------------------------------------
# TCN baseline
# ---------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    """
    Bloc résiduel pour le Temporal CNN.

    Conv1d causal (padding = (kernel-1)*dilation) + BatchNorm + ReLU + Dropout.
    Skip connection avec projection si les dimensions changent.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, channels, seq_len)
        residual = self.skip(x)

        out = self.conv1(x)[:, :, : -self.padding] if self.padding else self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.drop(out)

        out = self.conv2(out)[:, :, : -self.padding] if self.padding else self.conv2(out)
        out = self.relu(self.bn2(out))
        out = self.drop(out)

        return self.relu(out + residual)


class FireRiskTCN(nn.Module):
    """
    Temporal Convolutional Network — baseline pour la comparaison avec le LSTM.

    Empilement de TemporalBlocks avec dilatations exponentielles (1, 2, 4, 8).
    Global Average Pooling temporel → tête de régression → Sigmoid.

    Args:
        n_features  : features en entrée
        n_channels  : canaux dans les blocs TCN
        kernel_size : taille du noyau de convolution
        n_blocks    : nombre de blocs (dilatations = 2^0 … 2^(n_blocks-1))
        dropout     : taux de dropout
    """

    def __init__(
        self,
        n_features: int = 8,
        n_channels: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_ch = n_features
        for i in range(n_blocks):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, n_channels, kernel_size, dilation, dropout))
            in_ch = n_channels

        self.network = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(n_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, seq_len, n_features)
        Returns:
            out : (batch,) valeurs ∈ [0, 1]
        """
        # Conv1d attend (batch, channels, seq_len)
        x = x.permute(0, 2, 1)           # (batch, F, seq_len)
        out = self.network(x)             # (batch, n_channels, seq_len)
        pooled = out.mean(dim=-1)         # (batch, n_channels)
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    model_type: str = "lstm",
    n_features: int = 8,
    **kwargs,
) -> nn.Module:
    """
    Instancie et initialise le modèle demandé.

    Args:
        model_type : 'lstm' ou 'tcn'
        n_features : nombre de features d'entrée
        **kwargs   : hyperparamètres passés au constructeur
    Returns:
        modèle PyTorch prêt à l'entraînement
    """
    if model_type == "lstm":
        model = FireRiskLSTM(n_features=n_features, **kwargs)
        model.init_weights()
    elif model_type == "tcn":
        model = FireRiskTCN(n_features=n_features, **kwargs)
    else:
        raise ValueError(f"model_type inconnu : '{model_type}'. Choisir 'lstm' ou 'tcn'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modèle {model_type.upper()} : {n_params:,} paramètres entraînables")
    return model


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """Vérifie les deux architectures avec un batch fictif."""
    torch.manual_seed(42)
    batch_size, seq_len, n_features = 32, 15, 8
    x = torch.randn(batch_size, seq_len, n_features)

    for mtype in ("lstm", "tcn"):
        model = build_model(mtype, n_features=n_features)
        out = model(x)
        assert out.shape == (batch_size,), f"Forme incorrecte : {out.shape}"
        assert out.min() >= 0.0 and out.max() <= 1.0, "Sortie hors [0,1]"
        print(f"{mtype.upper()} OK — out.shape={out.shape}  min={out.min():.4f}  max={out.max():.4f}")


if __name__ == "__main__":
    main()
