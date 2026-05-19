"""
pipeline.py — Real-data pipeline for Mediterranean fire risk prediction.

Split temporel strict :
  Train : 2022-01-01 → 2022-12-15
  Val   : 2022-12-16 → 2022-12-31
  Test  : 2023-01-01 → 2023-12-31

Scaler fitté uniquement sur les données Train.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes géographiques
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 35.0, 47.0
LON_MIN, LON_MAX = -5.0, 28.0
RES = 0.25

LATS = np.arange(LAT_MIN, LAT_MAX + RES / 2, RES).astype(np.float32)
LONS = np.arange(LON_MIN, LON_MAX + RES / 2, RES).astype(np.float32)
N_LAT, N_LON = len(LATS), len(LONS)

# Splits temporels
TRAIN_START = pd.Timestamp("2022-01-01")
TRAIN_END   = pd.Timestamp("2022-12-15")
VAL_START   = pd.Timestamp("2022-12-16")
VAL_END     = pd.Timestamp("2022-12-31")
TEST_START  = pd.Timestamp("2023-01-01")
TEST_END    = pd.Timestamp("2023-12-31")

SEQ_LEN          = 15
FORECAST_HORIZON = 7

FEATURE_NAMES = [
    "ndvi", "temp_max", "humidity", "wind_speed",
    "precip", "drought_index", "past_hotspot_count", "past_frp",
]


# ---------------------------------------------------------------------------
# Helpers géographiques
# ---------------------------------------------------------------------------

def lat_to_idx(lat: float) -> int:
    return int(round((lat - LAT_MIN) / RES))


def lon_to_idx(lon: float) -> int:
    return int(round((lon - LON_MIN) / RES))


# ---------------------------------------------------------------------------
# Calculs physiques
# ---------------------------------------------------------------------------

def compute_relative_humidity(t2m: np.ndarray, d2m: np.ndarray) -> np.ndarray:
    gamma_t  = 17.625 * t2m / (243.04 + t2m + 1e-6)
    gamma_td = 17.625 * d2m / (243.04 + d2m + 1e-6)
    rh = 100.0 * np.exp(gamma_td) / (np.exp(gamma_t) + 1e-6)
    return np.clip(rh, 0.0, 100.0).astype(np.float32)


def compute_drought_index(temp_arr: np.ndarray, precip_arr: np.ndarray) -> np.ndarray:
    di = np.clip((temp_arr - 20.0) / 20.0 - precip_arr / 10.0, 0.0, 1.0)
    return di.astype(np.float32)


# ---------------------------------------------------------------------------
# Remplissage des NaN temporels (interpolation linéaire par cellule)
# ---------------------------------------------------------------------------

def fill_temporal_nan(arr: np.ndarray) -> np.ndarray:
    """Interpole les NaN le long de l'axe temporel pour chaque cellule (lat, lon)."""
    T, NL, NG = arr.shape
    flat = arr.reshape(T, NL * NG)
    df = pd.DataFrame(flat.astype(np.float64))
    df = df.interpolate(method="linear", limit_direction="both", axis=0)
    df = df.ffill(axis=0).bfill(axis=0)
    return df.values.reshape(T, NL, NG).astype(np.float32)


def fill_spatial_nan(arr: np.ndarray) -> np.ndarray:
    """Propage les valeurs terre vers les cellules mer (NaN permanent) via ffill/bfill spatial."""
    T, NL, NG = arr.shape
    result = arr.copy()
    for t in range(T):
        frame = result[t]
        if not np.isnan(frame).any():
            continue
        df = pd.DataFrame(frame.astype(np.float64))
        df = df.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)
        result[t] = df.values.astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# FIRMS — chargement vectorisé
# ---------------------------------------------------------------------------

def load_firms_year(firms_dir: Path, year: int, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Charge les CSV FIRMS pour une année. Retourne (hotspots, frp) de shape (T, N_LAT, N_LON)."""
    T = len(dates)
    hotspot_arr = np.zeros((T, N_LAT, N_LON), dtype=np.float32)
    frp_arr     = np.zeros((T, N_LAT, N_LON), dtype=np.float32)

    csv_files = list(firms_dir.glob("*.csv")) if firms_dir.exists() else []
    if not csv_files:
        logger.warning(f"Aucun CSV FIRMS dans {firms_dir} → hotspots = 0")
        return hotspot_arr, frp_arr

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "frp", "confidence"])
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Erreur lecture {f.name}: {e}")
            try:
                df = pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "frp"])
                dfs.append(df)
            except Exception:
                pass

    if not dfs:
        return hotspot_arr, frp_arr

    all_firms = pd.concat(dfs, ignore_index=True)
    all_firms["acq_date"] = pd.to_datetime(all_firms["acq_date"])
    # Filtrer sur l'année cible
    all_firms = all_firms[all_firms["acq_date"].dt.year == year].copy()

    if all_firms.empty:
        return hotspot_arr, frp_arr

    # Snap vers la grille 0.25°
    all_firms["lat_idx"] = ((all_firms["latitude"]  - LAT_MIN) / RES).round().astype(int)
    all_firms["lon_idx"] = ((all_firms["longitude"] - LON_MIN) / RES).round().astype(int)
    mask = (
        (all_firms["lat_idx"] >= 0) & (all_firms["lat_idx"] < N_LAT) &
        (all_firms["lon_idx"] >= 0) & (all_firms["lon_idx"] < N_LON)
    )
    all_firms = all_firms[mask]

    # Index date → position dans le tableau
    date_to_idx = {d.date(): i for i, d in enumerate(dates)}

    for date_val, day_df in all_firms.groupby("acq_date"):
        d = date_val.date() if hasattr(date_val, "date") else pd.Timestamp(date_val).date()
        if d not in date_to_idx:
            continue
        t_idx = date_to_idx[d]
        for (li, lj), grp in day_df.groupby(["lat_idx", "lon_idx"]):
            hotspot_arr[t_idx, li, lj] = len(grp)
            frp_arr[t_idx, li, lj]     = float(grp["frp"].mean())

    logger.info(f"FIRMS {year}: {all_firms.shape[0]:,} détections chargées")
    return hotspot_arr, frp_arr


# ---------------------------------------------------------------------------
# NDVI — chargement GeoTIFF + interpolation journalière
# ---------------------------------------------------------------------------

def load_ndvi_year(
    ndvi_dir: Path,
    year: int,
    dates: pd.DatetimeIndex,
    proxy_year: int | None = None,
) -> np.ndarray:
    """
    Charge les GeoTIFF NDVI pour une année.
    Si proxy_year est fourni, réutilise les TIF de proxy_year avec mapping DOY→DOY.
    Interpolation linéaire pour remplir les jours manquants.
    """
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        import rasterio.transform as rt
    except ImportError:
        logger.warning("rasterio indisponible → NDVI simulé")
        return _simulate_ndvi_series(dates)

    T = len(dates)
    ndvi_arr = np.full((T, N_LAT, N_LON), np.nan, dtype=np.float32)

    tif_year = proxy_year if proxy_year else year
    ndvi_files = sorted(ndvi_dir.glob(f"*NDVI*doy{tif_year}*.tif"))
    if not ndvi_files:
        ndvi_files = sorted(ndvi_dir.glob(f"*EVI*doy{tif_year}*.tif"))
    if not ndvi_files:
        logger.warning(f"Aucun fichier NDVI/EVI pour {tif_year} → NDVI simulé")
        return _simulate_ndvi_series(dates)

    dst_transform = rt.from_bounds(
        float(LONS[0]),  float(LATS[0]),
        float(LONS[-1]) + RES, float(LATS[-1]) + RES,
        N_LON, N_LAT,
    )

    doy_to_grid: dict[int, np.ndarray] = {}

    for tif_file in ndvi_files:
        fname = tif_file.name
        marker = f"doy{tif_year}"
        if marker not in fname:
            continue
        idx = fname.index(marker) + len(marker)
        try:
            doy = int(fname[idx:idx + 3])
        except ValueError:
            continue

        try:
            with rasterio.open(tif_file) as src:
                data = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
                data[(data < -3000) | (data > 10000)] = np.nan

                dst = np.full((N_LAT, N_LON), np.nan, dtype=np.float32)
                reproject(
                    source=data,
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.average,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                )
            # Facteur d'échelle MODIS
            dst = dst * 0.0001
            dst[(dst < -0.1) | (dst > 1.0)] = np.nan
            doy_to_grid[doy] = dst
        except Exception as e:
            logger.warning(f"Erreur lecture {tif_file.name}: {e}")

    if not doy_to_grid:
        logger.warning("Aucun NDVI valide → NDVI simulé")
        return _simulate_ndvi_series(dates)

    # Assigner le NDVI au premier jour correspondant à chaque DOY
    for i, date in enumerate(dates):
        target_doy = date.dayofyear if proxy_year is None else date.dayofyear
        if target_doy in doy_to_grid:
            ndvi_arr[i] = doy_to_grid[target_doy]
        elif proxy_year:
            # Trouver le DOY le plus proche dans les données proxy
            closest_doy = min(doy_to_grid.keys(), key=lambda d: abs(d - target_doy))
            ndvi_arr[i] = doy_to_grid[closest_doy]

    logger.info(f"NDVI {tif_year} → {len(doy_to_grid)} observations, interpolation en cours...")
    ndvi_arr = fill_temporal_nan(ndvi_arr)
    ndvi_arr = fill_spatial_nan(ndvi_arr)
    return ndvi_arr


def _simulate_ndvi_series(dates: pd.DatetimeIndex) -> np.ndarray:
    T = len(dates)
    arr = np.zeros((T, N_LAT, N_LON), dtype=np.float32)
    rng = np.random.default_rng(42)
    for i, d in enumerate(dates):
        phase = np.sin(2 * np.pi * d.dayofyear / 365)
        base = 0.35 + 0.15 * phase
        arr[i] = np.clip(base + rng.normal(0, 0.05, (N_LAT, N_LON)), -0.1, 1.0)
    return arr


# ---------------------------------------------------------------------------
# ERA5 — chargement NetCDF
# ---------------------------------------------------------------------------

def _open_nc_from_zip_or_file(nc_file: Path, var_name: str) -> "xr.Dataset | None":
    """
    Ouvre un fichier ERA5 qui peut être :
      - un NetCDF4 classique
      - un ZIP contenant des .nc séparés par variable
    Retourne le Dataset pour var_name, ou None si introuvable.
    """
    import zipfile
    import io
    import xarray as xr

    # Vérifier si c'est un ZIP (magic bytes PK)
    with open(nc_file, "rb") as fh:
        magic = fh.read(4)
    is_zip = magic[:2] == b"PK"

    if is_zip:
        # Mapping : mot-clé dans le nom de fichier ZIP → variable ERA5
        VAR_MAP = {
            "2m_temperature":        "t2m",
            "2m_dewpoint":           "d2m",
            "10m_u_component":       "u10",
            "10m_v_component":       "v10",
            "total_precipitation":   "tp",
        }
        with zipfile.ZipFile(nc_file) as z:
            for zname in z.namelist():
                for keyword, v in VAR_MAP.items():
                    if keyword in zname and v == var_name:
                        with z.open(zname) as f:
                            data = f.read()
                        try:
                            return xr.open_dataset(io.BytesIO(data), engine="h5netcdf")
                        except Exception:
                            return xr.open_dataset(io.BytesIO(data), engine="scipy")
        return None
    else:
        try:
            ds = xr.open_dataset(nc_file, engine="netcdf4")
        except Exception:
            try:
                ds = xr.open_dataset(nc_file, engine="h5netcdf")
            except Exception:
                ds = xr.open_dataset(nc_file, engine="scipy")
        if var_name in ds:
            return ds
        ds.close()
        return None


def load_era5_year(era5_dir: Path, year: int, dates: pd.DatetimeIndex) -> dict[str, np.ndarray] | None:
    """Charge les données ERA5 mensuelles pour une année. Retourne dict d'arrays (T, N_LAT, N_LON)."""
    try:
        import xarray as xr
    except ImportError:
        logger.error("xarray non installé")
        return None

    nc_files = sorted(era5_dir.glob(f"era5_med_{year}_*.nc"))
    if not nc_files:
        logger.warning(f"Aucun fichier ERA5 pour {year} dans {era5_dir}")
        return None

    T = len(dates)
    date_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}

    t2m_arr = np.full((T, N_LAT, N_LON), np.nan, dtype=np.float32)
    d2m_arr = np.full((T, N_LAT, N_LON), np.nan, dtype=np.float32)
    u10_arr = np.zeros((T, N_LAT, N_LON), dtype=np.float32)
    v10_arr = np.zeros((T, N_LAT, N_LON), dtype=np.float32)
    tp_arr  = np.zeros((T, N_LAT, N_LON), dtype=np.float32)

    logger.info(f"Chargement ERA5 {year}: {len(nc_files)} fichiers...")

    def _extract_var(ds: "xr.Dataset", var_name: str, times, factor: float = 1.0, offset: float = 0.0):
        """Extrait une variable d'un dataset et remplit les arrays cibles."""
        if ds is None or var_name not in ds:
            return None
        time_dim = next(
            (td for td in ["valid_time", "time", "forecast_reference_time"] if td in ds.dims),
            None,
        )
        if time_dim is None:
            return None
        # Latitude croissante
        lat_vals = ds.latitude.values
        if float(lat_vals[0]) > float(lat_vals[-1]):
            ds = ds.isel(latitude=slice(None, None, -1))
        # ERA5-Land a des NaN sur les cellules mer — propager les valeurs terra voisines
        # avant interpolation pour éviter la contamination des cellules côtières
        ds = ds.ffill(dim="latitude").bfill(dim="latitude")
        ds = ds.ffill(dim="longitude").bfill(dim="longitude")
        # Interpolation sur grille cible
        ds_i = ds.interp(latitude=LATS.astype(float), longitude=LONS.astype(float), method="linear")
        vals = ds_i[var_name].values.astype(np.float32) * factor + offset
        file_times = pd.to_datetime(ds_i[time_dim].values)
        ds.close()
        ds_i.close()
        return vals, file_times

    for nc_file in nc_files:
        try:
            n_loaded = 0
            for var, arr, factor, offset in [
                ("t2m", t2m_arr, 1.0, -273.15),
                ("d2m", d2m_arr, 1.0, -273.15),
                ("u10", u10_arr, 1.0,  0.0),
                ("v10", v10_arr, 1.0,  0.0),
                ("tp",  tp_arr,  1000.0, 0.0),
            ]:
                ds = _open_nc_from_zip_or_file(nc_file, var)
                if ds is None:
                    continue
                result = _extract_var(ds, var, None, factor, offset)
                if result is None:
                    continue
                vals, file_times = result
                for i_t, t in enumerate(file_times):
                    ts = pd.Timestamp(t).normalize()
                    if ts in date_to_idx:
                        idx = date_to_idx[ts]
                        if var == "tp":
                            arr[idx] = np.clip(vals[i_t], 0, None)
                        else:
                            arr[idx] = vals[i_t]
                n_loaded += 1

            logger.info(f"  {nc_file.name} → {n_loaded}/5 variables")

        except Exception as e:
            logger.warning(f"Erreur {nc_file.name}: {e}")
            continue

    # Interpoler les jours manquants
    if np.isnan(t2m_arr).any():
        logger.info("Interpolation des jours ERA5 manquants...")
        t2m_arr = fill_temporal_nan(t2m_arr)
        d2m_arr = fill_temporal_nan(d2m_arr)

    humidity   = compute_relative_humidity(t2m_arr, d2m_arr)
    wind_speed = np.sqrt(u10_arr ** 2 + v10_arr ** 2).astype(np.float32)

    return {
        "temp_max":   t2m_arr,
        "d2m":        d2m_arr,
        "humidity":   humidity,
        "wind_speed": wind_speed,
        "precip":     tp_arr,
    }


def _simulate_era5_year(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    T = len(dates)
    rng = np.random.default_rng(42)
    shape = (T, N_LAT, N_LON)
    temp_arr  = np.zeros(shape, dtype=np.float32)
    hum_arr   = np.zeros(shape, dtype=np.float32)
    wind_arr  = np.zeros(shape, dtype=np.float32)
    prec_arr  = np.zeros(shape, dtype=np.float32)
    for i, d in enumerate(dates):
        summer = max(0.0, float(np.sin(np.pi * (d.dayofyear - 80) / 180)))
        temp_base = 20 + 15 * summer
        temp_arr[i] = rng.normal(temp_base, 5, (N_LAT, N_LON)).astype(np.float32)
        hum_arr[i]  = np.clip(rng.normal(60 - 20 * summer, 10, (N_LAT, N_LON)), 10, 100).astype(np.float32)
        wind_arr[i] = rng.exponential(3, (N_LAT, N_LON)).astype(np.float32)
        prec_arr[i] = rng.exponential(0.5 * (1 - summer), (N_LAT, N_LON)).astype(np.float32)
    d2m = temp_arr - 5
    return {
        "temp_max":   temp_arr,
        "d2m":        d2m,
        "humidity":   hum_arr,
        "wind_speed": wind_arr,
        "precip":     prec_arr,
    }


# ---------------------------------------------------------------------------
# Risk score
# ---------------------------------------------------------------------------

def compute_risk_score(hotspot_arr: np.ndarray, horizon: int = FORECAST_HORIZON) -> np.ndarray:
    """
    Compte les hotspots dans un rayon 1 cellule (fenêtre 3×3) sur t+1 → t+horizon.
    Normalise par le percentile 99 des valeurs positives.
    """
    T, NL, NG = hotspot_arr.shape
    risk = np.zeros((T, NL, NG), dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)

    for t in range(T - horizon):
        future_sum = hotspot_arr[t + 1 : t + 1 + horizon].sum(axis=0)
        risk[t] = convolve(future_sum, kernel, mode="reflect")

    pos_vals = risk[risk > 0]
    if len(pos_vals) > 0:
        p99 = float(np.percentile(pos_vals, 99))
        if p99 > 0:
            risk = risk / p99

    return np.clip(risk, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Construction du DataFrame annuel
# ---------------------------------------------------------------------------

def build_year_dataframe(
    dates: pd.DatetimeIndex,
    era5: dict[str, np.ndarray],
    hotspot_arr: np.ndarray,
    frp_arr: np.ndarray,
    ndvi_arr: np.ndarray,
) -> pd.DataFrame:
    """Assemble un DataFrame (T × N_LAT × N_LON, features) pour une année."""
    T = len(dates)
    assert T == era5["temp_max"].shape[0] == hotspot_arr.shape[0] == ndvi_arr.shape[0]

    temp_arr   = era5["temp_max"]
    precip_arr = era5["precip"]
    hum_arr    = era5["humidity"]
    wind_arr   = era5["wind_speed"]

    drought_arr = compute_drought_index(temp_arr, precip_arr)
    risk_arr    = compute_risk_score(hotspot_arr)

    # Coordonnées meshgrid
    dates_arr = np.repeat(dates.values[:, None, None], N_LAT, axis=1)
    dates_arr = np.repeat(dates_arr, N_LON, axis=2)      # (T, NL, NG)
    lats_arr  = np.tile(LATS[None, :, None], (T, 1, N_LON))  # (T, NL, NG)
    lons_arr  = np.tile(LONS[None, None, :], (T, N_LAT, 1))  # (T, NL, NG)

    # Indices lat/lon répétés sur T jours : shape (T * N_LAT * N_LON,)
    lat_idx_arr = np.tile(np.repeat(np.arange(N_LAT), N_LON), T)
    lon_idx_arr = np.tile(np.arange(N_LON), T * N_LAT)

    df = pd.DataFrame({
        "date":               dates_arr.ravel(),
        "lat":                lats_arr.ravel().round(2),
        "lon":                lons_arr.ravel().round(2),
        "lat_idx":            lat_idx_arr,
        "lon_idx":            lon_idx_arr,
        "ndvi":               ndvi_arr.ravel(),
        "temp_max":           temp_arr.ravel(),
        "humidity":           hum_arr.ravel(),
        "wind_speed":         wind_arr.ravel(),
        "precip":             precip_arr.ravel(),
        "drought_index":      drought_arr.ravel(),
        "past_hotspot_count": hotspot_arr.ravel(),
        "past_frp":           frp_arr.ravel(),
        "risk_score":         risk_arr.ravel(),
    })

    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(raw_dir: str | Path, out_dir: str | Path) -> None:
    """
    Traitement complet 2022 + 2023.
    Sauvegarde parquet, .npy (train/val/test), scaler.pkl, grid_meta.json.
    """
    import time as _time

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = _time.time()

    # ── 2022 ──
    logger.info("═══ TRAITEMENT 2022 ═══")
    dates_2022 = pd.date_range("2022-01-01", "2022-12-31", freq="D")

    era5_2022 = load_era5_year(raw_dir / "era5", 2022, dates_2022)
    if era5_2022 is None:
        logger.warning("ERA5 2022 absent → simulation")
        era5_2022 = _simulate_era5_year(dates_2022)

    hotspot_2022, frp_2022 = load_firms_year(raw_dir / "firms", 2022, dates_2022)
    ndvi_2022 = load_ndvi_year(raw_dir / "ndvi", 2022, dates_2022)

    df_2022 = build_year_dataframe(dates_2022, era5_2022, hotspot_2022, frp_2022, ndvi_2022)
    df_2022.to_parquet(out_dir / "dataset_2022.parquet", index=False)
    logger.info(f"dataset_2022.parquet sauvegardé ({len(df_2022):,} lignes)")

    # ── 2023 ──
    logger.info("═══ TRAITEMENT 2023 ═══")
    dates_2023 = pd.date_range("2023-01-01", "2023-12-31", freq="D")

    era5_dir_2023 = raw_dir / "era5_2023"
    has_era5_2023 = era5_dir_2023.exists() and len(list(era5_dir_2023.glob("*.nc"))) > 0

    if has_era5_2023:
        era5_2023 = load_era5_year(era5_dir_2023, 2023, dates_2023)
        if era5_2023 is None:
            era5_2023 = _simulate_era5_year(dates_2023)
    else:
        logger.warning("ERA5 2023 absent → simulation climatique")
        era5_2023 = _simulate_era5_year(dates_2023)

    firms_dir_2023 = raw_dir / "firms_2023"
    has_firms_2023 = firms_dir_2023.exists() and len(list(firms_dir_2023.glob("*.csv"))) > 0
    hotspot_2023, frp_2023 = load_firms_year(
        firms_dir_2023 if has_firms_2023 else raw_dir / "firms_empty_placeholder",
        2023, dates_2023,
    )

    # NDVI 2023 : proxy DOY depuis 2022
    logger.info("NDVI 2023 : utilisation proxy 2022 (même DOY)")
    ndvi_2023 = load_ndvi_year(raw_dir / "ndvi", 2022, dates_2023, proxy_year=2022)

    df_2023 = build_year_dataframe(dates_2023, era5_2023, hotspot_2023, frp_2023, ndvi_2023)
    df_2023.to_parquet(out_dir / "dataset_2023.parquet", index=False)
    logger.info(f"dataset_2023.parquet sauvegardé ({len(df_2023):,} lignes)")

    # ── 2024 ──
    logger.info("═══ TRAITEMENT 2024 ═══")
    dates_2024 = pd.date_range("2024-01-01", "2024-12-31", freq="D")

    era5_dir_2024 = raw_dir / "era5_2024"
    has_era5_2024 = era5_dir_2024.exists() and len(list(era5_dir_2024.glob("*.nc"))) > 0
    if has_era5_2024:
        era5_2024 = load_era5_year(era5_dir_2024, 2024, dates_2024)
        if era5_2024 is None:
            era5_2024 = _simulate_era5_year(dates_2024)
    else:
        logger.warning("ERA5 2024 absent → simulation climatique")
        era5_2024 = _simulate_era5_year(dates_2024)

    firms_dir_2024 = raw_dir / "firms_2024"
    has_firms_2024 = firms_dir_2024.exists() and len(list(firms_dir_2024.glob("*.csv"))) > 0
    hotspot_2024, frp_2024 = load_firms_year(
        firms_dir_2024 if has_firms_2024 else raw_dir / "firms_empty_placeholder",
        2024, dates_2024,
    )

    logger.info("NDVI 2024 : utilisation proxy 2022 (même DOY)")
    ndvi_2024 = load_ndvi_year(raw_dir / "ndvi", 2022, dates_2024, proxy_year=2022)

    df_2024 = build_year_dataframe(dates_2024, era5_2024, hotspot_2024, frp_2024, ndvi_2024)
    df_2024.to_parquet(out_dir / "dataset_2024.parquet", index=False)
    logger.info(f"dataset_2024.parquet sauvegardé ({len(df_2024):,} lignes)")

    # ── Reconstruction des arrays (T, N_LAT, N_LON, F) ──
    logger.info("═══ CONSTRUCTION ARRAYS NUMPY ═══")

    def df_to_arrays(df_year: pd.DataFrame, dates_sel: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        sub = df_year[df_year["date"].isin(dates_sel)].sort_values(["date", "lat_idx", "lon_idx"])
        T_sub = len(dates_sel)
        X = sub[FEATURE_NAMES].values.reshape(T_sub, N_LAT, N_LON, len(FEATURE_NAMES)).astype(np.float32)
        y = sub["risk_score"].values.reshape(T_sub, N_LAT, N_LON).astype(np.float32)
        return X, y

    # Train : Jan 1 → Dec 15 2022 (349 jours)
    train_dates = pd.date_range(TRAIN_START, TRAIN_END, freq="D")
    X_train, y_train = df_to_arrays(df_2022, train_dates)
    logger.info(f"Train  : X{X_train.shape}  y{y_train.shape}")

    # Val : Dec 2 → Dec 31 2022 (30 jours — inclut 14 jours de contexte)
    val_dates = pd.date_range("2022-12-02", VAL_END, freq="D")
    X_val, y_val = df_to_arrays(df_2022, val_dates)
    logger.info(f"Val    : X{X_val.shape}  y{y_val.shape}")

    # Test : Dec 18 2022 → Dec 31 2023 (14 jours de contexte + 365 jours test)
    ctx_dates  = pd.date_range("2022-12-18", "2022-12-31", freq="D")
    test_dates = pd.date_range(TEST_START, TEST_END, freq="D")
    X_ctx, y_ctx = df_to_arrays(df_2022, ctx_dates)
    X_test_year, y_test_year = df_to_arrays(df_2023, test_dates)
    X_test = np.concatenate([X_ctx, X_test_year], axis=0)
    y_test = np.concatenate([y_ctx, y_test_year], axis=0)
    logger.info(f"Test   : X{X_test.shape}  y{y_test.shape}")

    # ── Scaler — fitté uniquement sur Train ──
    logger.info("Calcul du scaler (train uniquement)...")
    X_flat = X_train.reshape(-1, len(FEATURE_NAMES))
    scaler = {
        "min_": X_flat.min(axis=0),
        "max_": X_flat.max(axis=0),
    }
    scaler["range_"] = np.where(
        scaler["max_"] - scaler["min_"] > 0,
        scaler["max_"] - scaler["min_"],
        1.0,
    )

    def scale(X):
        return ((X - scaler["min_"]) / scaler["range_"]).astype(np.float32)

    X_train_s = scale(X_train)
    X_val_s   = scale(X_val)
    X_test_s  = scale(X_test)

    # ── Sauvegarde ──
    np.save(out_dir / "X_train.npy", X_train_s)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy",   X_val_s)
    np.save(out_dir / "y_val.npy",   y_val)
    np.save(out_dir / "X_test.npy",  X_test_s)
    np.save(out_dir / "y_test.npy",  y_test)
    logger.info("Arrays .npy sauvegardés")

    # Sauvegarde du scaler (format compatible avec dataset.FeatureScaler)
    scaler_pkl = {"min_": scaler["min_"], "max_": scaler["max_"]}
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler_pkl, f)
    logger.info("scaler.pkl sauvegardé")

    # grid_meta.json
    meta = {
        "lat_min": LAT_MIN, "lat_max": LAT_MAX,
        "lon_min": LON_MIN, "lon_max": LON_MAX,
        "resolution": RES,
        "n_lat": N_LAT, "n_lon": N_LON,
        "features": FEATURE_NAMES,
        "seq_len": SEQ_LEN,
        "has_era5_2023": has_era5_2023,
        "has_firms_2023": has_firms_2023,
    }
    with open(out_dir / "grid_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = _time.time() - t0
    logger.info(f"Pipeline terminé en {elapsed:.1f}s")
    logger.info(f"Fichiers dans {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        logger.info(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    base = Path(__file__).resolve().parents[1]
    run_pipeline(base / "data" / "raw", base / "data" / "processed")


if __name__ == "__main__":
    main()
