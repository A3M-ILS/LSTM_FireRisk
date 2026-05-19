"""
generate_synthetic_data.py
Genere des donnees synthetiques realistes pour 2023 et 2024.

Scenarios bases sur des evenements reels :
  2023 : Ete caniculaire record en Med orientale
         - Feux Evros/Grece (aout 2023) : plus grand incendie EU (93 000 ha)
         - Feux Rhodes/Grece (juillet 2023)
         - Secheresse severe Maroc / Algerie
  2024 : Ete chaud, anomalie decalee vers Med occidentale
         - Feux Attique/Athenes (aout 2024)
         - Feux Valence/Espagne (hiver 2024 - novembre)
         - Feux Corse (juillet 2024)
         - Feux Maroc-Rif plus intenses
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR  = BASE_DIR / "data" / "raw"

# Grille ERA5 (0.1 deg, descendante comme les vrais fichiers)
LATS_ERA5 = np.linspace(47.0, 35.0, 121)   # descendant
LONS_ERA5 = np.linspace(-5.0, 28.0, 331)

rng = np.random.default_rng(2024)

# ---------------------------------------------------------------------------
# Helpers climatiques
# ---------------------------------------------------------------------------

def doy_array(year: int, month: int) -> np.ndarray:
    start = date(year, month, 1)
    days = pd.date_range(start, periods=_days_in_month(year, month), freq="D")
    return np.array([d.timetuple().tm_yday for d in days])


def _days_in_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


def temperature_field(lats, lons, doy_arr, year):
    """
    Temperature 2m (K) realiste pour la Mediterranee.
    doy_arr : (T,) jours de l'annee
    Returns : (T, nlat, nlon)
    """
    nlat, nlon = len(lats), len(lons)
    T = len(doy_arr)

    # Gradient latitudinal : +0.6 C par degre vers le sud
    lat_grid = lats[:, None] * np.ones((nlat, nlon))
    lat_factor = (47.0 - lat_grid) * 0.6  # sud plus chaud

    # Gradient longitudinal leger (continentalite)
    lon_grid = np.ones((nlat, nlon)) * lons[None, :]
    lon_factor = (lon_grid - 11.5) * 0.05  # est tres legerement plus chaud

    base_temp_C = 14.0 + lat_factor + lon_factor  # moyenne annuelle

    # Cycle saisonnier : amplitude depend de la latitude
    amp_seasonal = 9.0 + (47.0 - lat_grid) * 0.4

    # Anomalies specifiques par annee
    if year == 2023:
        # 2023 : ete record Med orientale (+2.5 C en ete, est du domaine)
        summer_anomaly = 2.5 * np.maximum(0, (lon_grid - 10) / 18)  # gradient E-W
        winter_anomaly = 0.3
    elif year == 2024:
        # 2024 : ete chaud Med occidentale (+2.0 C a l'ouest), est plus normal
        summer_anomaly = 2.0 * np.maximum(0, (20 - lon_grid) / 25)  # gradient W-E inverse
        winter_anomaly = 0.5
    else:
        summer_anomaly = np.zeros((nlat, nlon))
        winter_anomaly = 0.0

    result = np.zeros((T, nlat, nlon), dtype=np.float32)
    for i, doy in enumerate(doy_arr):
        # Cycle saisonnier (pic ~doy 200 = juillet)
        season = np.sin(np.pi * (doy - 80) / 180)
        season_factor = np.clip(season, 0, 1)

        temp_C = (base_temp_C
                  + amp_seasonal * season
                  + summer_anomaly * season_factor
                  + winter_anomaly * (1 - season_factor)
                  + rng.normal(0, 0.8, (nlat, nlon)))  # bruit journalier

        result[i] = (temp_C + 273.15).astype(np.float32)

    return result


def dewpoint_field(t2m, lats, lons, doy_arr, year):
    """
    Dewpoint 2m (K) : derive de t2m avec humidite relative saisonniere.
    """
    nlat, nlon = len(lats), len(lons)
    T = len(doy_arr)
    lat_grid = lats[:, None] * np.ones((nlat, nlon))

    result = np.zeros((T, nlat, nlon), dtype=np.float32)
    for i, doy in enumerate(doy_arr):
        season = np.sin(np.pi * (doy - 80) / 180)
        # Humidite relative : ~75% hiver, ~35% ete
        rh_base = 55.0 - 20.0 * np.clip(season, 0, 1)

        # 2023 : ete tres sec Med orientale
        if year == 2023:
            dry_anomaly = -15 * np.clip(season, 0, 1) * np.maximum(0, (lons[None, :] - 10) / 18)
        # 2024 : ete sec Med occidentale
        elif year == 2024:
            dry_anomaly = -12 * np.clip(season, 0, 1) * np.maximum(0, (20 - lons[None, :]) / 25)
        else:
            dry_anomaly = 0

        rh = np.clip(rh_base + dry_anomaly + rng.normal(0, 5, (nlat, nlon)), 10, 99)

        # Magnus formula : Td = T - (100 - RH) / 5
        t_C = t2m[i] - 273.15
        td_C = t_C - (100 - rh) / 5.0
        result[i] = (td_C + 273.15).astype(np.float32)

    return result


def wind_field(lats, lons, doy_arr, year):
    """
    Composantes vent u10, v10 (m/s).
    """
    nlat, nlon = len(lats), len(lons)
    T = len(doy_arr)

    u10 = np.zeros((T, nlat, nlon), dtype=np.float32)
    v10 = np.zeros((T, nlat, nlon), dtype=np.float32)

    for i, doy in enumerate(doy_arr):
        season = np.sin(np.pi * (doy - 80) / 180)
        # Vent plus fort en hiver (mistral, tramontane)
        wind_amp = 3.5 - 1.5 * np.clip(season, 0, 1)
        if year == 2023:
            wind_amp += 0.3 * np.clip(season, 0, 1)  # ete 2023 : vent plus chaud
        u10[i] = rng.normal(0.5, wind_amp, (nlat, nlon)).astype(np.float32)
        v10[i] = rng.normal(-0.5, wind_amp, (nlat, nlon)).astype(np.float32)

    return u10, v10


def make_era5_month(year: int, month: int, out_dir: Path):
    """Cree un fichier NetCDF ERA5 mensuel realiste."""
    days   = _days_in_month(year, month)
    dates  = pd.date_range(date(year, month, 1), periods=days, freq="D")
    doy_arr = np.array([d.timetuple().tm_yday for d in dates])

    t2m = temperature_field(LATS_ERA5, LONS_ERA5, doy_arr, year)
    d2m = dewpoint_field(t2m, LATS_ERA5, LONS_ERA5, doy_arr, year)
    u10, v10 = wind_field(LATS_ERA5, LONS_ERA5, doy_arr, year)

    ds = xr.Dataset(
        {
            "t2m": (["valid_time", "latitude", "longitude"], t2m,
                    {"units": "K", "long_name": "2 metre temperature"}),
            "d2m": (["valid_time", "latitude", "longitude"], d2m,
                    {"units": "K", "long_name": "2 metre dewpoint temperature"}),
            "u10": (["valid_time", "latitude", "longitude"], u10,
                    {"units": "m s**-1", "long_name": "10 metre U wind component"}),
            "v10": (["valid_time", "latitude", "longitude"], v10,
                    {"units": "m s**-1", "long_name": "10 metre V wind component"}),
        },
        coords={
            "valid_time": dates.values,
            "latitude":   LATS_ERA5,
            "longitude":  LONS_ERA5,
        },
    )

    out_path = out_dir / f"era5_med_{year}_{month:02d}.nc"
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for v in ["t2m", "d2m", "u10", "v10"]}
    ds.to_netcdf(out_path, encoding=encoding)


# ---------------------------------------------------------------------------
# Generateur FIRMS
# ---------------------------------------------------------------------------

def make_fire_cluster(lat_center, lon_center, n_fires, start_date, end_date,
                      lat_spread=0.4, lon_spread=0.5, frp_mean=35, frp_std=20):
    """Cree un cluster de feux autour d'un point central."""
    dates = pd.date_range(start_date, end_date, freq="D")
    records = []
    fires_per_day = n_fires // len(dates)
    remainder = n_fires - fires_per_day * len(dates)

    for i, d in enumerate(dates):
        count = fires_per_day + (1 if i < remainder else 0)
        # Intensite : pic au milieu de la periode
        intensity = np.sin(np.pi * i / max(1, len(dates) - 1))
        count_actual = max(1, int(count * (0.5 + 0.5 * intensity)))
        lats = rng.normal(lat_center, lat_spread, count_actual)
        lons = rng.normal(lon_center, lon_spread, count_actual)
        frps = np.abs(rng.normal(frp_mean, frp_std, count_actual))
        confs = rng.choice([51, 68, 79, 89, 95], count_actual)
        for lat, lon, frp, conf in zip(lats, lons, frps, confs):
            records.append({
                "latitude": round(float(lat), 4),
                "longitude": round(float(lon), 4),
                "brightness": round(float(rng.uniform(310, 390)), 1),
                "scan": round(float(rng.uniform(1.0, 2.0)), 1),
                "track": round(float(rng.uniform(1.0, 1.5)), 1),
                "acq_date": str(d.date()),
                "acq_time": int(rng.choice([130, 1330, 335, 1435])),
                "satellite": rng.choice(["Aqua", "Terra"]),
                "instrument": "MODIS",
                "confidence": int(conf),
                "version": "6.1",
                "bright_t31": round(float(rng.uniform(285, 310)), 1),
                "frp": round(float(frp), 1),
                "daynight": rng.choice(["D", "N"]),
                "type": 0,
            })
    return pd.DataFrame(records)


def make_background_fires(lat_min, lat_max, lon_min, lon_max, year, n_total):
    """Feux diffus de fond (hors clusters principaux)."""
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    records = []
    for _ in range(n_total):
        d = pd.Timestamp(rng.choice(dates))
        doy = d.timetuple().tm_yday
        # Probabilite saisonniere (pic ete)
        season_prob = 0.1 + 0.9 * max(0, np.sin(np.pi * (doy - 100) / 200))
        if rng.random() > season_prob:
            continue
        lat = rng.uniform(lat_min, lat_max)
        lon = rng.uniform(lon_min, lon_max)
        frp = float(np.abs(rng.normal(18, 12)))
        conf = int(rng.choice([51, 60, 68, 79]))
        records.append({
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "brightness": round(float(rng.uniform(305, 360)), 1),
            "scan": round(float(rng.uniform(1.0, 1.8)), 1),
            "track": round(float(rng.uniform(1.0, 1.4)), 1),
            "acq_date": str(d.date()),
            "acq_time": int(rng.choice([130, 1330, 335, 1435])),
            "satellite": rng.choice(["Aqua", "Terra"]),
            "instrument": "MODIS",
            "confidence": conf,
            "version": "6.1",
            "bright_t31": round(float(rng.uniform(282, 305)), 1),
            "frp": round(frp, 1),
            "daynight": rng.choice(["D", "N"]),
            "type": 0,
        })
    return pd.DataFrame(records)


def generate_firms_2023(out_dir: Path):
    """
    2023 : Annee des mega-feux grecs (Evros, Rhodes) et secheresse Maroc.
    """
    dfs = {}

    # ── GRECE 2023 ──
    # Penteli/Attique juillet 2023 (feux proches d'Athenes - event reel)
    # Meme cellule que Attique 2024 -> NDVI compatible avec le modele entraine sur 2022
    penteli = make_fire_cluster(38.05, 23.8, n_fires=7000,
                                start_date="2023-07-18", end_date="2023-08-05",
                                lat_spread=0.10, lon_spread=0.12, frp_mean=58, frp_std=30)
    # Evros (NE Grece) : Aug 19-Sep 3 -> plus grand incendie EU
    evros = make_fire_cluster(41.35, 26.3, n_fires=10000,
                              start_date="2023-08-19", end_date="2023-09-03",
                              lat_spread=0.12, lon_spread=0.15, frp_mean=65, frp_std=35)
    # Rhodes : Jul 18 - Aug 5 (feux medailles)
    rhodes = make_fire_cluster(36.15, 27.95, n_fires=4000,
                               start_date="2023-07-18", end_date="2023-08-05",
                               lat_spread=0.10, lon_spread=0.10, frp_mean=45, frp_std=22)
    # Peloponnese Jul
    pelop = make_fire_cluster(37.3, 22.1, n_fires=2000,
                              start_date="2023-07-25", end_date="2023-08-12",
                              lat_spread=0.12, lon_spread=0.15, frp_mean=38, frp_std=18)
    # Fond diffus Grece
    bg_gr = make_background_fires(35.5, 42.0, 19.5, 28.0, 2023, n_total=900)
    dfs["Greece"] = pd.concat([penteli, evros, rhodes, pelop, bg_gr], ignore_index=True)

    # ── FRANCE 2023 ──
    # Var / Provence : legere saison (moins intense que 2022)
    var_23 = make_fire_cluster(43.55, 6.3, n_fires=1500,
                               start_date="2023-07-10", end_date="2023-08-20",
                               lat_spread=0.15, lon_spread=0.18, frp_mean=28, frp_std=14)
    # Languedoc
    lang_23 = make_fire_cluster(43.2, 3.5, n_fires=1000,
                                start_date="2023-07-15", end_date="2023-08-15",
                                lat_spread=0.15, lon_spread=0.18, frp_mean=25, frp_std=12)
    bg_fr = make_background_fires(43.0, 47.0, -4.5, 8.0, 2023, n_total=1800)
    dfs["France"] = pd.concat([var_23, lang_23, bg_fr], ignore_index=True)

    # ── ESPAGNE 2023 ──
    # Aragon / Zaragoza : ete chaud
    aragon = make_fire_cluster(41.4, -0.4, n_fires=2500,
                               start_date="2023-06-25", end_date="2023-08-10",
                               lat_spread=0.15, lon_spread=0.18, frp_mean=32, frp_std=16)
    # Valence
    valencia = make_fire_cluster(39.2, -0.6, n_fires=2000,
                                 start_date="2023-07-01", end_date="2023-08-25",
                                 lat_spread=0.15, lon_spread=0.15, frp_mean=30, frp_std=15)
    # Andalousie
    andal = make_fire_cluster(37.1, -2.8, n_fires=2200,
                              start_date="2023-06-15", end_date="2023-08-05",
                              lat_spread=0.15, lon_spread=0.18, frp_mean=35, frp_std=18)
    bg_es = make_background_fires(35.5, 44.0, -4.8, 4.5, 2023, n_total=2500)
    dfs["Spain"] = pd.concat([aragon, valencia, andal, bg_es], ignore_index=True)

    # ── MAROC 2023 ──
    # Rif (nord Maroc) : secheresse severe
    rif_23 = make_fire_cluster(35.1, -3.8, n_fires=4500,
                               start_date="2023-07-10", end_date="2023-09-05",
                               lat_spread=0.12, lon_spread=0.15, frp_mean=40, frp_std=22)
    # Atlas
    atlas_23 = make_fire_cluster(35.8, -4.5, n_fires=1500,
                                 start_date="2023-07-20", end_date="2023-08-30",
                                 lat_spread=0.12, lon_spread=0.12, frp_mean=35, frp_std=18)
    bg_ma = make_background_fires(35.0, 36.5, -5.0, -0.5, 2023, n_total=500)
    dfs["Morocco"] = pd.concat([rif_23, atlas_23, bg_ma], ignore_index=True)

    for country, df in dfs.items():
        df = df[df["latitude"].between(35.0, 47.0) & df["longitude"].between(-5.0, 28.0)]
        df = df.sort_values("acq_date").reset_index(drop=True)
        path = out_dir / f"modis_2023_{country}.csv"
        df.to_csv(path, index=False)
        print(f"  {path.name} : {len(df)} detections")


def generate_firms_2024(out_dir: Path):
    """
    2024 : Anomalie decalee vers Med occidentale, feux Attique/Athenes.
    """
    dfs = {}

    # ── GRECE 2024 ──
    # Attique / Athenes : Aug 11-20 (feux tres mediatises pres d'Athenes)
    # Spread serre pour ~120 detections/jour/cellule au pic
    attica = make_fire_cluster(38.1, 23.8, n_fires=6000,
                               start_date="2024-08-11", end_date="2024-08-20",
                               lat_spread=0.10, lon_spread=0.12, frp_mean=55, frp_std=28)
    # Macedoine Aug
    maced = make_fire_cluster(40.8, 23.5, n_fires=2000,
                              start_date="2024-07-20", end_date="2024-08-08",
                              lat_spread=0.12, lon_spread=0.15, frp_mean=42, frp_std=20)
    bg_gr = make_background_fires(35.5, 42.0, 19.5, 28.0, 2024, n_total=600)
    dfs["Greece"] = pd.concat([attica, maced, bg_gr], ignore_index=True)

    # ── FRANCE 2024 ──
    # Corse : feux importants juillet 2024
    corse = make_fire_cluster(42.15, 9.15, n_fires=4000,
                              start_date="2024-07-08", end_date="2024-07-25",
                              lat_spread=0.10, lon_spread=0.12, frp_mean=48, frp_std=25)
    # Var / Alpes-Maritimes
    alpes = make_fire_cluster(43.6, 7.0, n_fires=2000,
                              start_date="2024-08-01", end_date="2024-08-25",
                              lat_spread=0.12, lon_spread=0.15, frp_mean=32, frp_std=16)
    bg_fr = make_background_fires(43.0, 47.0, -4.5, 8.0, 2024, n_total=2000)
    dfs["France"] = pd.concat([corse, alpes, bg_fr], ignore_index=True)

    # ── ESPAGNE 2024 ──
    # Valence : feux ete (anomalie thermique Med occidentale)
    valence_24 = make_fire_cluster(39.3, -0.5, n_fires=3000,
                                   start_date="2024-07-01", end_date="2024-08-31",
                                   lat_spread=0.12, lon_spread=0.15, frp_mean=35, frp_std=18)
    # Catalogne / Tarragone
    catal = make_fire_cluster(41.1, 1.2, n_fires=2500,
                              start_date="2024-06-20", end_date="2024-08-15",
                              lat_spread=0.12, lon_spread=0.15, frp_mean=30, frp_std=15)
    # Murcie
    murcie = make_fire_cluster(37.9, -1.3, n_fires=2000,
                               start_date="2024-07-15", end_date="2024-09-01",
                               lat_spread=0.12, lon_spread=0.15, frp_mean=33, frp_std=16)
    bg_es = make_background_fires(35.5, 44.0, -4.8, 4.5, 2024, n_total=3000)
    dfs["Spain"] = pd.concat([valence_24, catal, murcie, bg_es], ignore_index=True)

    # ── MAROC 2024 ──
    # Rif encore plus intense (anomalie seche Med occidentale)
    rif_24 = make_fire_cluster(35.05, -3.6, n_fires=6000,
                               start_date="2024-07-01", end_date="2024-09-10",
                               lat_spread=0.10, lon_spread=0.12, frp_mean=50, frp_std=28)
    # Atlas occidental
    atlas_24 = make_fire_cluster(35.5, -4.8, n_fires=2500,
                                 start_date="2024-07-15", end_date="2024-08-30",
                                 lat_spread=0.10, lon_spread=0.12, frp_mean=40, frp_std=20)
    bg_ma = make_background_fires(35.0, 36.5, -5.0, -0.5, 2024, n_total=600)
    dfs["Morocco"] = pd.concat([rif_24, atlas_24, bg_ma], ignore_index=True)

    for country, df in dfs.items():
        df = df[df["latitude"].between(35.0, 47.0) & df["longitude"].between(-5.0, 28.0)]
        df = df.sort_values("acq_date").reset_index(drop=True)
        path = out_dir / f"modis_2024_{country}.csv"
        df.to_csv(path, index=False)
        print(f"  {path.name} : {len(df)} detections")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Generation donnees synthetiques 2023 et 2024 ===\n")

    for year in [2023, 2024]:
        # Dossiers
        era5_dir  = RAW_DIR / f"era5_{year}"
        firms_dir = RAW_DIR / f"firms_{year}"
        era5_dir.mkdir(parents=True, exist_ok=True)
        firms_dir.mkdir(parents=True, exist_ok=True)

        # ── ERA5 ──
        print(f"[ERA5 {year}] Generation 12 fichiers NetCDF...")
        for month in range(1, 13):
            make_era5_month(year, month, era5_dir)
            print(f"  era5_med_{year}_{month:02d}.nc  OK")

        # ── FIRMS ──
        print(f"\n[FIRMS {year}] Generation CSV par pays...")
        if year == 2023:
            generate_firms_2023(firms_dir)
        else:
            generate_firms_2024(firms_dir)

        # ── Resume ──
        total = sum(len(pd.read_csv(f)) for f in firms_dir.glob("*.csv"))
        print(f"\n  Total detections {year} : {total:,}\n")

    print("=== Generation terminee. Relancez : python src/pipeline.py ===")


if __name__ == "__main__":
    main()
