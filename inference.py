"""
inference.py — Prédiction ponctuelle du risque d'incendie.

Usage :
    python src/inference.py --lat 43.3 --lon 5.4 --date 2023-08-15 --model lstm
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Ajouter le répertoire src/ au path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Constantes (doivent correspondre à pipeline.py)
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 35.0, 47.0
LON_MIN, LON_MAX = -5.0, 28.0
RES = 0.25

LATS = np.arange(LAT_MIN, LAT_MAX + RES / 2, RES).astype(np.float32)
LONS = np.arange(LON_MIN, LON_MAX + RES / 2, RES).astype(np.float32)
N_LAT, N_LON = len(LATS), len(LONS)

SEQ_LEN = 15
FEATURE_NAMES = [
    "ndvi", "temp_max", "humidity", "wind_speed",
    "precip", "drought_index", "past_hotspot_count", "past_frp",
]

LEVELS = [
    (0.0,  0.2, "Faible",   "#4CAF50"),
    (0.2,  0.4, "Modéré",   "#FFC107"),
    (0.4,  0.6, "Élevé",    "#FF9800"),
    (0.6,  1.0, "Critique", "#F44336"),
]


# ---------------------------------------------------------------------------
# Helpers géographiques
# ---------------------------------------------------------------------------

def lat_to_idx(lat: float) -> int:
    idx = int(round((lat - LAT_MIN) / RES))
    return max(0, min(N_LAT - 1, idx))


def lon_to_idx(lon: float) -> int:
    idx = int(round((lon - LON_MIN) / RES))
    return max(0, min(N_LON - 1, idx))


def risk_level(score: float) -> tuple[str, str]:
    for lo, hi, label, color in LEVELS:
        if lo <= score < hi or (score >= 0.6 and hi == 1.0):
            return label, color
    return "Critique", "#F44336"


# ---------------------------------------------------------------------------
# Chargement modèle + scaler
# ---------------------------------------------------------------------------

def load_model(model_type: str, processed_dir: Path) -> torch.nn.Module:
    from model import build_model

    ckpt_path = processed_dir / f"best_model_{model_type}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")

    model = build_model(model_type, n_features=len(FEATURE_NAMES))
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_scaler(processed_dir: Path) -> dict:
    import pickle

    scaler_path = processed_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler.pkl introuvable : {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def scale_features(X: np.ndarray, scaler: dict) -> np.ndarray:
    """Normalise X avec le scaler fitté sur train. X : (seq_len, n_features)."""
    min_  = scaler["min_"]
    max_  = scaler["max_"]
    range_ = np.where(max_ - min_ > 0, max_ - min_, 1.0)
    return ((X - min_) / range_).astype(np.float32)


# ---------------------------------------------------------------------------
# Extraction de la fenêtre de features
# ---------------------------------------------------------------------------

def get_feature_window(
    lat: float,
    lon: float,
    date: str,
    processed_dir: Path,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, list[str], bool]:
    """
    Extrait la fenêtre de seq_len jours précédant `date` pour la cellule (lat, lon).

    Returns:
        (window, dates_list, used_fallback)
        window : (seq_len, n_features) array, non normalisé
    """
    target_date = pd.Timestamp(date)
    li = lat_to_idx(lat)
    lj = lon_to_idx(lon)

    # Choisir le dataset approprié
    parquet_2023 = processed_dir / "dataset_2023.parquet"
    parquet_2022 = processed_dir / "dataset_2022.parquet"

    used_fallback = False

    parquet_2024 = processed_dir / "dataset_2024.parquet"

    if target_date.year == 2024 and parquet_2024.exists():
        df = pd.read_parquet(parquet_2024)
        df["date"] = pd.to_datetime(df["date"])
    elif target_date.year == 2023 and parquet_2023.exists():
        df = pd.read_parquet(parquet_2023)
        df["date"] = pd.to_datetime(df["date"])
    elif parquet_2022.exists():
        df = pd.read_parquet(parquet_2022)
        df["date"] = pd.to_datetime(df["date"])
        # Fallback : remplacer annee par 2022
        if target_date.year != 2022:
            target_date = target_date.replace(year=2022)
            used_fallback = True
    else:
        # Fallback parquet global (ancien format)
        old_parquet = processed_dir / "fire_risk_dataset.parquet"
        if old_parquet.exists():
            df = pd.read_parquet(old_parquet)
            df["date"] = pd.to_datetime(df["date"])
            used_fallback = True
        else:
            raise FileNotFoundError("Aucun dataset parquet trouvé. Lance d'abord pipeline.py.")

    # Filtrer sur la cellule
    cell_df = df[(df["lat_idx"] == li) & (df["lon_idx"] == lj)].copy()

    if cell_df.empty:
        # Chercher la cellule la plus proche
        actual_lat = round(LAT_MIN + li * RES, 2)
        actual_lon = round(LON_MIN + lj * RES, 2)
        cell_df = df[
            (df["lat"].round(2) == actual_lat) &
            (df["lon"].round(2) == actual_lon)
        ].copy()

    if cell_df.empty:
        raise ValueError(
            f"Aucune donnée pour la cellule lat={lat:.2f}, lon={lon:.2f} "
            f"(idx lat={li}, lon={lj})"
        )

    cell_df = cell_df.sort_values("date").reset_index(drop=True)

    # Trouver la position de la date cible dans le dataset
    dates_in_df = cell_df["date"].values
    target_ts   = pd.Timestamp(target_date)

    # Chercher la date exacte ou la plus proche
    date_diffs = np.abs((pd.DatetimeIndex(dates_in_df) - target_ts).days)
    closest_idx = int(np.argmin(date_diffs))

    if date_diffs[closest_idx] > 7:
        raise ValueError(
            f"Date {date} trop éloignée des données disponibles "
            f"(plus proche : {pd.Timestamp(dates_in_df[closest_idx]).date()})"
        )

    # Extraire la fenêtre de seq_len jours se terminant à closest_idx
    start_idx = closest_idx - seq_len + 1

    if start_idx < 0:
        available = closest_idx + 1
        raise ValueError(
            f"Historique insuffisant : {available} jour(s) disponibles avant cette date, "
            f"minimum requis : {seq_len}. "
            f"Choisissez une date après le {pd.Timestamp(dates_in_df[seq_len-1]).date()}."
        )

    window_df   = cell_df.iloc[start_idx:closest_idx + 1]
    window_vals = window_df[FEATURE_NAMES].values.astype(np.float32)
    dates_list  = [str(pd.Timestamp(d).date()) for d in window_df["date"].values]

    return window_vals, dates_list, used_fallback


# ---------------------------------------------------------------------------
# Prédiction principale
# ---------------------------------------------------------------------------

def predict(
    lat: float,
    lon: float,
    date: str,
    model_type: str = "lstm",
    processed_dir: Optional[Path] = None,
) -> dict:
    """
    Prédit le risque d'incendie pour un point géographique et une date.

    Returns dict avec risk_score, niveau, couleur, features, série temporelle.
    """
    if processed_dir is None:
        processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"

    # Vérification géographique
    if not (LAT_MIN <= lat <= LAT_MAX):
        raise ValueError(f"Latitude {lat} hors zone [{LAT_MIN}, {LAT_MAX}]")
    if not (LON_MIN <= lon <= LON_MAX):
        raise ValueError(f"Longitude {lon} hors zone [{LON_MIN}, {LON_MAX}]")

    # Charger modèle et scaler
    model  = load_model(model_type, processed_dir)
    scaler = load_scaler(processed_dir)

    # Extraire la fenêtre
    window, dates_list, used_fallback = get_feature_window(
        lat, lon, date, processed_dir
    )

    # Normaliser
    window_norm = scale_features(window, scaler)

    # Inférence
    x_tensor = torch.from_numpy(window_norm).unsqueeze(0)  # (1, seq_len, F)
    with torch.no_grad():
        score = float(model(x_tensor).item())
    score = float(np.clip(score, 0.0, 1.0))

    niveau, couleur = risk_level(score)

    # Statistiques sur la fenêtre (valeurs dénormalisées = window original)
    feature_stats = {
        "ndvi_moyen":         float(np.nanmean(window[:, 0])),
        "temp_max_moyenne":   float(np.nanmean(window[:, 1])),
        "humidity_moyenne":   float(np.nanmean(window[:, 2])),
        "wind_speed_moyen":   float(np.nanmean(window[:, 3])),
        "hotspots_passes":    int(window[:, 6].sum()),
    }

    # Prédictions journalières sur la fenêtre
    preds_jour = []
    for i in range(1, len(dates_list) + 1):
        sub = window_norm[:i]
        if len(sub) < SEQ_LEN:
            preds_jour.append(None)
            continue
        sub_t = torch.from_numpy(sub[-SEQ_LEN:]).unsqueeze(0)
        with torch.no_grad():
            preds_jour.append(float(np.clip(model(sub_t).item(), 0, 1)))

    result = {
        "risk_score":         round(score, 4),
        "niveau":             niveau,
        "couleur":            couleur,
        "cellule":            {
            "lat": float(LATS[lat_to_idx(lat)]),
            "lon": float(LONS[lon_to_idx(lon)]),
        },
        "date_prediction":    date,
        "modele":             model_type.upper(),
        "fallback_2022":      used_fallback,
        "features_utilisees": feature_stats,
        "serie_temporelle":   {
            "dates":        dates_list,
            "ndvi":         [round(float(v), 4) for v in window[:, 0]],
            "temp_max":     [round(float(v), 2) for v in window[:, 1]],
            "humidity":     [round(float(v), 2) for v in window[:, 2]],
            "wind_speed":   [round(float(v), 2) for v in window[:, 3]],
            "risk_predit":  [round(p, 4) if p is not None else None for p in preds_jour],
        },
    }
    return result


# ---------------------------------------------------------------------------
# Prédiction en lot (grille complète pour une date)
# ---------------------------------------------------------------------------

def predict_grid(
    date: str,
    model_type: str = "lstm",
    processed_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Prédit le risque pour toutes les cellules de la grille à une date donnée.
    Retourne array (N_LAT, N_LON) de risk_scores ∈ [0, 1].
    """
    if processed_dir is None:
        processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"

    target_date = pd.Timestamp(date)

    # Charger modèle et scaler
    model  = load_model(model_type, processed_dir)
    scaler = load_scaler(processed_dir)

    # Chercher les données dans le bon parquet
    parquet_2023 = processed_dir / "dataset_2023.parquet"
    parquet_2022 = processed_dir / "dataset_2022.parquet"

    if target_date.year == 2023 and parquet_2023.exists():
        df = pd.read_parquet(parquet_2023)
    elif parquet_2022.exists():
        df = pd.read_parquet(parquet_2022)
        if target_date.year != 2022:
            target_date = target_date.replace(year=2022)
    else:
        return np.zeros((N_LAT, N_LON), dtype=np.float32)

    df["date"] = pd.to_datetime(df["date"])

    # Fenêtre de SEQ_LEN jours se terminant à target_date
    start_date = target_date - pd.Timedelta(days=SEQ_LEN - 1)
    window_df  = df[
        (df["date"] >= start_date) & (df["date"] <= target_date)
    ].sort_values(["date", "lat_idx", "lon_idx"])

    if window_df.empty or window_df["date"].nunique() < SEQ_LEN:
        return np.zeros((N_LAT, N_LON), dtype=np.float32)

    # Construire X: (N_cells, SEQ_LEN, F)
    n_cells = N_LAT * N_LON
    X = np.zeros((n_cells, SEQ_LEN, len(FEATURE_NAMES)), dtype=np.float32)

    dates_in_win = sorted(window_df["date"].unique())[-SEQ_LEN:]
    for t_idx, d in enumerate(dates_in_win):
        day_df = window_df[window_df["date"] == d].sort_values(["lat_idx", "lon_idx"])
        if len(day_df) == n_cells:
            X[:, t_idx, :] = day_df[FEATURE_NAMES].values.astype(np.float32)

    # Normaliser
    min_  = scaler["min_"]
    max_  = scaler["max_"]
    range_ = np.where(max_ - min_ > 0, max_ - min_, 1.0)
    X_norm = ((X - min_) / range_).astype(np.float32)

    # Inférence par batch
    BATCH = 2048
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_cells, BATCH):
            end   = min(start + BATCH, n_cells)
            x_b   = torch.from_numpy(X_norm[start:end])
            preds.append(model(x_b).cpu().numpy())

    pred_arr = np.clip(np.concatenate(preds), 0, 1).astype(np.float32)
    return pred_arr.reshape(N_LAT, N_LON)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prédiction risque incendie ponctuelle")
    parser.add_argument("--lat",   type=float, default=43.3,        help="Latitude")
    parser.add_argument("--lon",   type=float, default=5.4,         help="Longitude")
    parser.add_argument("--date",  type=str,   default="2023-08-15",help="Date YYYY-MM-DD")
    parser.add_argument("--model", type=str,   default="lstm",      choices=["lstm", "tcn"])
    args = parser.parse_args()

    print(f"\n[FIRE] Prediction risque incendie")
    print(f"   Lat={args.lat}, Lon={args.lon}, Date={args.date}, Modele={args.model.upper()}")
    print("-" * 50)

    try:
        result = predict(args.lat, args.lon, args.date, model_type=args.model)

        print(f"   Risk score : {result['risk_score']:.4f}")
        print(f"   Niveau     : {result['niveau']}  {result['couleur']}")
        print(f"   Cellule    : lat={result['cellule']['lat']:.2f}, lon={result['cellule']['lon']:.2f}")

        if result["fallback_2022"]:
            print("   [!] Donnees 2023 absentes, utilisation 2022 en proxy")

        print("\n   Features utilisees (15 jours) :")
        for k, v in result["features_utilisees"].items():
            print(f"     {k:25s} : {v}")

        print("\n   Serie temporelle (derniers 5 jours) :")
        series = result["serie_temporelle"]
        for i in range(-5, 0):
            r = series["risk_predit"][i]
            r_str = f"{r:.3f}" if r is not None else "  N/A"
            print(
                f"     {series['dates'][i]}  "
                f"T={series['temp_max'][i]:.1f}C  "
                f"NDVI={series['ndvi'][i]:.3f}  "
                f"Risque={r_str}"
            )

        print("\n[OK] Resultat JSON complet disponible via predict()")

    except FileNotFoundError as e:
        print(f"[ERR] Fichier manquant : {e}")
        print("   Lance d'abord : python src/pipeline.py && python src/train.py")
    except ValueError as e:
        print(f"[ERR] {e}")
    except Exception as e:
        import traceback
        print(f"[ERR] Erreur inattendue : {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
