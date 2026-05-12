"""
pipeline.py — Fusion des trois sources de données pour le Fire Risk Project.

Sources :
  - MODIS NDVI    : GeoTIFF 1 km, 16 jours  → reprojection sur grille 0.25°
  - ERA5          : NetCDF journalier         → temp, humidity, wind, precip
  - NASA FIRMS    : CSV hotspots              → comptage par cellule/jour

Sortie : Parquet + NumPy .npy prêts pour le Dataset PyTorch.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes géographiques
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 35.0, 47.0   # °N
LON_MIN, LON_MAX = -5.0, 28.0   # °E
RES = 0.25                       # résolution de la grille commune

LATS = np.arange(LAT_MIN, LAT_MAX + RES, RES)
LONS = np.arange(LON_MIN, LON_MAX + RES, RES)
N_LAT, N_LON = len(LATS), len(LONS)

# Splits temporels stricts
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END   = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END   = "2023-12-31"

FORECAST_HORIZON = 7   # jours futurs pour construire le label
SEQ_LEN = 15           # longueur de séquence LSTM


# ---------------------------------------------------------------------------
# Helpers géographiques
# ---------------------------------------------------------------------------

def lat_to_idx(lat: float) -> int:
    """Convertit une latitude en index de ligne dans la grille 0.25°."""
    return int(round((lat - LAT_MIN) / RES))


def lon_to_idx(lon: float) -> int:
    """Convertit une longitude en index de colonne dans la grille 0.25°."""
    return int(round((lon - LON_MIN) / RES))


# ---------------------------------------------------------------------------
# Calculs physiques
# ---------------------------------------------------------------------------

def compute_relative_humidity(t2m: np.ndarray, d2m: np.ndarray) -> np.ndarray:
    """
    Calcule l'humidité relative (%) via la formule de Magnus.

    RH = 100 * exp(17.625*Td/(243.04+Td)) / exp(17.625*T/(243.04+T))

    Args:
        t2m : température 2 m en °C
        d2m : température du point de rosée 2 m en °C
    Returns:
        humidité relative clampée dans [0, 100]
    """
    gamma_t  = 17.625 * t2m  / (243.04 + t2m)
    gamma_td = 17.625 * d2m  / (243.04 + d2m)
    rh = 100.0 * np.exp(gamma_td) / np.exp(gamma_t)
    return np.clip(rh, 0.0, 100.0)


def compute_wind_speed(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    """Calcule la vitesse du vent (m/s) depuis les composantes u et v."""
    return np.sqrt(u10 ** 2 + v10 ** 2)


# ---------------------------------------------------------------------------
# Ingestion NDVI (GeoTIFF)
# ---------------------------------------------------------------------------

def load_ndvi_geotiff(tiff_path: Path, target_lats: np.ndarray, target_lons: np.ndarray) -> np.ndarray:
    """
    Lit un GeoTIFF NDVI 1 km et le reprojette/rééchantillonne sur la grille 0.25°.

    Args:
        tiff_path    : chemin vers le fichier .tif
        target_lats  : tableau de latitudes cibles
        target_lons  : tableau de longitudes cibles
    Returns:
        ndarray shape (n_lat, n_lon) avec valeurs NDVI ∈ [-1, 1]
    """
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        import rasterio.transform as rt

        with rasterio.open(tiff_path) as src:
            data = src.read(1).astype(np.float32)
            src_nodata = src.nodata
            if src_nodata is not None:
                data[data == src_nodata] = np.nan

            # Construire la transform cible 0.25°
            dst_transform = rt.from_bounds(
                LON_MIN, LAT_MIN, LON_MAX + RES, LAT_MAX + RES,
                len(target_lons), len(target_lats)
            )
            dst_crs = "EPSG:4326"
            dst = np.full((len(target_lats), len(target_lons)), np.nan, dtype=np.float32)

            reproject(
                source=data,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average,
            )
        # Mise à l'échelle MODIS : diviser par 10 000
        dst = dst / 10_000.0
        return np.clip(dst, -1.0, 1.0)

    except ImportError:
        logger.warning("rasterio non disponible — NDVI simulé.")
        return _simulate_ndvi(target_lats, target_lons)


def _simulate_ndvi(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Simule un champ NDVI réaliste pour la Méditerranée."""
    rng = np.random.default_rng(42)
    n_lat, n_lon = len(lats), len(lons)
    base = 0.3 + 0.2 * np.sin(np.linspace(0, np.pi, n_lat))[:, None]
    noise = rng.normal(0, 0.05, (n_lat, n_lon))
    return np.clip(base + noise, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Ingestion ERA5 (NetCDF)
# ---------------------------------------------------------------------------

def load_era5_netcdf(nc_dir: Path, date: pd.Timestamp) -> dict[str, np.ndarray]:
    """
    Lit les champs ERA5 pour une date donnée depuis un répertoire de fichiers NetCDF.

    Variables extraites : t2m (°C), d2m (°C), u10, v10, tp (précipitations mm).
    Calcule ensuite humidity et wind_speed.

    Args:
        nc_dir : répertoire contenant les fichiers *.nc
        date   : date cible
    Returns:
        dict avec clés 'temp_max', 'humidity', 'wind_speed', 'precip'
    """
    try:
        import xarray as xr

        pattern = f"era5_med_{date.year}_{date.month:02d}.nc"
        nc_path = nc_dir / pattern
        if not nc_path.exists():
            logger.debug(f"ERA5 {pattern} absent → simulation.")
            return _simulate_era5()

        ds = xr.open_dataset(nc_path)
        sel = ds.sel(valid_time=date.date(), method="nearest") if "valid_time" in ds.dims else ds.isel(time=0)

        # Interpolation sur grille cible
        sel = sel.interp(latitude=LATS, longitude=LONS, method="linear")

        t2m = sel["t2m"].values - 273.15   # K → °C
        d2m = sel["d2m"].values - 273.15
        u10 = sel["u10"].values
        v10 = sel["v10"].values
        tp  = sel.get("tp", sel.get("tp_sum", None))
        precip = (tp.values * 1000.0) if tp is not None else np.zeros_like(t2m)

        humidity    = compute_relative_humidity(t2m, d2m)
        wind_speed  = compute_wind_speed(u10, v10)

        ds.close()
        return {
            "temp_max":   t2m.astype(np.float32),
            "humidity":   humidity.astype(np.float32),
            "wind_speed": wind_speed.astype(np.float32),
            "precip":     precip.astype(np.float32),
        }

    except Exception as exc:
        logger.warning(f"Erreur ERA5 ({exc}) → simulation.")
        return _simulate_era5()


def _simulate_era5() -> dict[str, np.ndarray]:
    """Simule des champs ERA5 pour une journée."""
    rng = np.random.default_rng(int(pd.Timestamp.now().timestamp()) % 10_000)
    shape = (N_LAT, N_LON)
    return {
        "temp_max":   rng.normal(25, 8, shape).astype(np.float32),
        "humidity":   rng.uniform(20, 90, shape).astype(np.float32),
        "wind_speed": rng.exponential(3, shape).astype(np.float32),
        "precip":     rng.exponential(0.5, shape).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Ingestion FIRMS (CSV)
# ---------------------------------------------------------------------------

def load_firms_csv(csv_dir: Path, date: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """
    Lit les CSV FIRMS et compte hotspots + FRP moyen par cellule de grille pour un jour donné.

    Format attendu des CSV : colonnes 'latitude', 'longitude', 'acq_date', 'frp'.

    Args:
        csv_dir : répertoire contenant les *.csv FIRMS
        date    : date cible
    Returns:
        Tuple (hotspot_count, frp_mean) de shape (N_LAT, N_LON)
    """
    date_str = date.strftime("%Y-%m-%d")
    hotspot_count = np.zeros((N_LAT, N_LON), dtype=np.float32)
    frp_mean      = np.zeros((N_LAT, N_LON), dtype=np.float32)

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        logger.debug(f"Aucun CSV FIRMS trouvé → simulation.")
        return _simulate_firms()

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "frp"])
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return _simulate_firms()

    df = pd.concat(dfs, ignore_index=True)
    day_df = df[df["acq_date"] == date_str]

    if day_df.empty:
        return hotspot_count, frp_mean

    # Snap vers la grille 0.25°
    day_df = day_df.copy()
    day_df["lat_idx"] = ((day_df["latitude"]  - LAT_MIN) / RES).round().astype(int)
    day_df["lon_idx"] = ((day_df["longitude"] - LON_MIN) / RES).round().astype(int)

    # Filtrer les points hors grille
    mask = (
        (day_df["lat_idx"] >= 0) & (day_df["lat_idx"] < N_LAT) &
        (day_df["lon_idx"] >= 0) & (day_df["lon_idx"] < N_LON)
    )
    day_df = day_df[mask]

    for (li, lj), grp in day_df.groupby(["lat_idx", "lon_idx"]):
        hotspot_count[li, lj] = len(grp)
        frp_mean[li, lj]      = grp["frp"].mean()

    return hotspot_count, frp_mean


def _simulate_firms() -> tuple[np.ndarray, np.ndarray]:
    """Simule des détections FIRMS rares (10% des cellules)."""
    rng = np.random.default_rng(int(pd.Timestamp.now().timestamp()) % 10_000)
    shape = (N_LAT, N_LON)
    mask = rng.random(shape) < 0.10
    counts = rng.integers(0, 5, shape).astype(np.float32) * mask
    frp    = rng.exponential(15, shape).astype(np.float32) * mask
    return counts, frp


# ---------------------------------------------------------------------------
# Construction du jeu de données fusionné
# ---------------------------------------------------------------------------

def build_risk_score(hotspot_series: np.ndarray, frp_series: np.ndarray, horizon: int = FORECAST_HORIZON) -> np.ndarray:
    """
    Construit le label risk_score ∈ [0, 1] sur un horizon futur de `horizon` jours.

    Stratégie : pour chaque cellule, le score est la somme pondérée des hotspots futurs
    (normalisée par la valeur max observée), avec un boost FRP.

    Args:
        hotspot_series : array (T, N_LAT, N_LON) de comptages hotspots
        frp_series     : array (T, N_LAT, N_LON) de FRP moyen
        horizon        : nombre de jours futurs à agréger
    Returns:
        risk_scores    : array (T, N_LAT, N_LON) clampé dans [0, 1]
    """
    T = hotspot_series.shape[0]
    risk = np.zeros_like(hotspot_series, dtype=np.float32)

    for t in range(T - horizon):
        future_hotspots = hotspot_series[t + 1 : t + 1 + horizon].sum(axis=0)
        future_frp      = frp_series[t + 1 : t + 1 + horizon].mean(axis=0)
        combined = future_hotspots + 0.01 * future_frp
        risk[t] = combined

    # Normalisation globale min-max
    max_val = risk.max()
    if max_val > 0:
        risk = risk / max_val

    return np.clip(risk, 0.0, 1.0)


def build_drought_index(temp_series: np.ndarray, precip_series: np.ndarray, window: int = 30) -> np.ndarray:
    """
    Calcule un indice de sécheresse simplifié : température cumulée / (précipitations cumulées + ε).

    Args:
        temp_series   : array (T, N_LAT, N_LON)
        precip_series : array (T, N_LAT, N_LON)
        window        : fenêtre glissante en jours
    Returns:
        drought_index normalisé dans [0, 1]
    """
    T = temp_series.shape[0]
    di = np.zeros_like(temp_series, dtype=np.float32)
    for t in range(T):
        start = max(0, t - window + 1)
        cum_temp   = temp_series[start:t + 1].sum(axis=0)
        cum_precip = precip_series[start:t + 1].sum(axis=0)
        di[t] = cum_temp / (cum_precip + 1.0)

    vmax = np.percentile(di, 99)
    if vmax > 0:
        di = di / vmax
    return np.clip(di, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_dir: str | Path,
    out_dir: str | Path,
    simulate: bool = True,
) -> pd.DataFrame:
    """
    Lance la fusion complète des trois sources et sauvegarde les artefacts.

    Args:
        raw_dir  : répertoire racine data/raw/
        out_dir  : répertoire data/processed/
        simulate : si True, génère des données synthétiques quand les vraies sont absentes
    Returns:
        DataFrame fusionné avec toutes les features et le label
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    firms_dir = raw_dir / "firms"
    ndvi_dir  = raw_dir / "ndvi"
    era5_dir  = raw_dir / "era5"

    # Plage de dates complète
    dates = pd.date_range("2018-01-01", "2023-12-31", freq="D")
    T = len(dates)

    logger.info(f"Traitement de {T} jours sur {N_LAT}×{N_LON} cellules …")

    # Pré-allocation des séries temporelles (T, N_LAT, N_LON)
    ndvi_series    = np.zeros((T, N_LAT, N_LON), dtype=np.float32)
    temp_series    = np.zeros((T, N_LAT, N_LON), dtype=np.float32)
    hum_series     = np.zeros((T, N_LAT, N_LON), dtype=np.float32)
    wind_series    = np.zeros((T, N_LAT, N_LON), dtype=np.float32)
    precip_series  = np.zeros((T, N_LAT, N_LON), dtype=np.float32)
    hotspot_series = np.zeros((T, N_LAT, N_LON), dtype=np.float32)
    frp_series     = np.zeros((T, N_LAT, N_LON), dtype=np.float32)

    rng_global = np.random.default_rng(42)   # reproductibilité

    ndvi_cache: dict[str, np.ndarray] = {}   # NDVI 16 jours → cache par période

    for i, date in enumerate(dates):
        if i % 100 == 0:
            logger.info(f"  → {date.date()} ({i}/{T})")

        # --- NDVI : mise à jour toutes les 16 jours ---
        period_key = f"{date.year}-{(date.dayofyear // 16):02d}"
        if period_key not in ndvi_cache:
            tiff_candidates = list(ndvi_dir.glob(f"*{date.year}*{date.month:02d}*.tif"))
            if tiff_candidates:
                ndvi_cache[period_key] = load_ndvi_geotiff(tiff_candidates[0], LATS, LONS)
            elif simulate:
                # Simulation saisonnière réaliste
                season_phase = np.sin(2 * np.pi * date.dayofyear / 365)
                base_ndvi = 0.35 + 0.15 * season_phase
                noise = rng_global.normal(0, 0.05, (N_LAT, N_LON)).astype(np.float32)
                ndvi_cache[period_key] = np.clip(base_ndvi + noise, -1.0, 1.0).astype(np.float32)
            else:
                ndvi_cache[period_key] = np.full((N_LAT, N_LON), np.nan, dtype=np.float32)

        ndvi_series[i] = ndvi_cache[period_key]

        # --- ERA5 ---
        if era5_dir.exists():
            era5 = load_era5_netcdf(era5_dir, date)
        elif simulate:
            # Simulation saisonnière réaliste
            summer = np.sin(np.pi * (date.dayofyear - 80) / 180)  # pic en juillet
            temp_base = 20 + 15 * max(0, summer)
            era5 = {
                "temp_max":   rng_global.normal(temp_base, 5, (N_LAT, N_LON)).astype(np.float32),
                "humidity":   rng_global.normal(60 - 20 * max(0, summer), 10, (N_LAT, N_LON)).clip(10, 100).astype(np.float32),
                "wind_speed": rng_global.exponential(3, (N_LAT, N_LON)).astype(np.float32),
                "precip":     rng_global.exponential(0.5 * (1 - max(0, summer)), (N_LAT, N_LON)).astype(np.float32),
            }
        else:
            era5 = {k: np.zeros((N_LAT, N_LON), dtype=np.float32) for k in ["temp_max", "humidity", "wind_speed", "precip"]}

        temp_series[i]   = era5["temp_max"]
        hum_series[i]    = era5["humidity"]
        wind_series[i]   = era5["wind_speed"]
        precip_series[i] = era5["precip"]

        # --- FIRMS ---
        if firms_dir.exists() and list(firms_dir.glob("*.csv")):
            hotspot_series[i], frp_series[i] = load_firms_csv(firms_dir, date)
        elif simulate:
            # Feux plus fréquents en été
            summer_factor = max(0, np.sin(np.pi * (date.dayofyear - 130) / 120))
            fire_prob = 0.03 + 0.12 * summer_factor
            mask = rng_global.random((N_LAT, N_LON)) < fire_prob
            hotspot_series[i] = rng_global.integers(0, 8, (N_LAT, N_LON)).astype(np.float32) * mask
            frp_series[i]     = rng_global.exponential(20, (N_LAT, N_LON)).astype(np.float32) * mask

    logger.info("Calcul du drought_index …")
    drought_series = build_drought_index(temp_series, precip_series)

    logger.info("Calcul du risk_score …")
    risk_series = build_risk_score(hotspot_series, frp_series, horizon=FORECAST_HORIZON)

    # --- Construction du DataFrame ---
    logger.info("Construction du DataFrame …")
    rows = []
    for i, date in enumerate(dates):
        for li in range(N_LAT):
            for lj in range(N_LON):
                rows.append({
                    "date":              date.date(),
                    "lat":               round(LATS[li], 2),
                    "lon":               round(LONS[lj], 2),
                    "ndvi":              ndvi_series[i, li, lj],
                    "temp_max":          temp_series[i, li, lj],
                    "humidity":          hum_series[i, li, lj],
                    "wind_speed":        wind_series[i, li, lj],
                    "precip":            precip_series[i, li, lj],
                    "drought_index":     drought_series[i, li, lj],
                    "past_hotspot_count": hotspot_series[i, li, lj],
                    "past_frp":          frp_series[i, li, lj],
                    "risk_score":        risk_series[i, li, lj],
                    "split":             _assign_split(date),
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # --- Sauvegarde Parquet ---
    parquet_path = out_dir / "fire_risk_dataset.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Parquet sauvegardé : {parquet_path}")

    # --- Sauvegarde NumPy pour accès rapide ---
    feature_names = ["ndvi", "temp_max", "humidity", "wind_speed",
                     "precip", "drought_index", "past_hotspot_count", "past_frp"]
    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split].sort_values(["date", "lat", "lon"])
        X = split_df[feature_names].values.reshape(-1, N_LAT, N_LON, len(feature_names))
        y = split_df["risk_score"].values.reshape(-1, N_LAT, N_LON)
        np.save(out_dir / f"X_{split}.npy", X)
        np.save(out_dir / f"y_{split}.npy", y)
        logger.info(f"NumPy {split} → X{X.shape} y{y.shape}")

    logger.info("Pipeline terminé.")
    return df


def _assign_split(date: pd.Timestamp) -> str:
    """Retourne 'train', 'val' ou 'test' selon le split temporel strict."""
    if date <= pd.Timestamp(TRAIN_END):
        return "train"
    elif date <= pd.Timestamp(VAL_END):
        return "val"
    else:
        return "test"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """Point d'entrée pour tester le pipeline en mode simulation."""
    import time

    base = Path(__file__).resolve().parents[1]
    raw_dir = base / "data" / "raw"
    out_dir = base / "data" / "processed"

    logger.info("=== TEST PIPELINE (mode simulé) ===")
    t0 = time.time()
    df = run_pipeline(raw_dir, out_dir, simulate=True)
    logger.info(f"Terminé en {time.time() - t0:.1f}s  — {len(df):,} lignes")
    print(df.head())
    print(df.describe())


if __name__ == "__main__":
    main()
