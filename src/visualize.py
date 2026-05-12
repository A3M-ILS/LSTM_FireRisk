"""
visualize.py — EDA et visualisations pour le Fire Risk Project.

Graphiques produits :
  1. Saisonnalité mensuelle des feux (hotspots)
  2. Distribution du NDVI par saison
  3. Matrice de corrélation des features
  4. Scatter risque vs température
  5. Carte de risque prédit sur la Méditerranée
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Grille méditerranéenne
LAT_MIN, LAT_MAX = 35.0, 47.0
LON_MIN, LON_MAX = -5.0, 28.0
RES = 0.25
LATS = np.arange(LAT_MIN, LAT_MAX + RES, RES)
LONS = np.arange(LON_MIN, LON_MAX + RES, RES)
N_LAT, N_LON = len(LATS), len(LONS)

FEATURE_COLS = [
    "ndvi", "temp_max", "humidity", "wind_speed",
    "precip", "drought_index", "past_hotspot_count", "past_frp",
]
SEASON_LABELS = {1: "Hiver", 2: "Hiver", 3: "Printemps", 4: "Printemps",
                 5: "Printemps", 6: "Été", 7: "Été", 8: "Été",
                 9: "Automne", 10: "Automne", 11: "Automne", 12: "Hiver"}
SEASON_PALETTE = {"Hiver": "#4e9af1", "Printemps": "#59b259",
                  "Été": "#f7941d", "Automne": "#c0392b"}


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_dataset(processed_dir: Path) -> Optional[pd.DataFrame]:
    """
    Charge le Parquet fusionné. Retourne None si absent.

    Args:
        processed_dir : répertoire data/processed/
    Returns:
        DataFrame ou None
    """
    parquet = processed_dir / "fire_risk_dataset.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"Dataset chargé : {len(df):,} lignes")
        return df
    logger.warning("Parquet absent → génération de données simulées pour l'EDA.")
    return None


def simulate_eda_data(n_cells: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Génère un DataFrame simulé réaliste pour l'EDA.

    Reproduit la saisonnalité méditerranéenne :
    - Feux concentrés en été (juillet–août)
    - NDVI bas en été, haut au printemps
    - Température haute en été

    Args:
        n_cells : nombre de cellules spatiales fictives
        seed    : graine aléatoire
    Returns:
        DataFrame avec colonnes compatibles avec le vrai dataset
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", "2023-12-31", freq="D")
    n_dates = len(dates)

    rows = []
    for d in dates:
        doy = d.dayofyear
        summer = max(0.0, np.sin(np.pi * (doy - 130) / 120))
        spring = max(0.0, np.sin(np.pi * (doy - 60)  / 120))

        for _ in range(n_cells // n_dates + 1):
            lat = rng.uniform(LAT_MIN, LAT_MAX)
            lon = rng.uniform(LON_MIN, LON_MAX)

            ndvi         = float(np.clip(0.55 + 0.2 * spring - 0.25 * summer + rng.normal(0, 0.08), -1, 1))
            temp_max     = float(10 + 22 * summer + rng.normal(0, 4))
            humidity     = float(np.clip(75 - 40 * summer + rng.normal(0, 8), 10, 100))
            wind_speed   = float(np.clip(rng.exponential(3) + 1, 0, 20))
            precip       = float(np.clip(rng.exponential(0.8 * (1 - 0.8 * summer)), 0, 30))
            drought_idx  = float(np.clip(0.2 + 0.7 * summer + rng.normal(0, 0.05), 0, 1))
            fire_prob    = 0.02 + 0.25 * summer
            is_fire      = rng.random() < fire_prob
            hotspot_cnt  = float(rng.integers(1, 8) if is_fire else 0)
            frp          = float(rng.exponential(20) if is_fire else 0)
            risk_score   = float(np.clip(
                0.05 + 0.6 * summer * is_fire + 0.2 * drought_idx + rng.normal(0, 0.03),
                0, 1
            ))

            rows.append({
                "date": d, "lat": round(lat, 2), "lon": round(lon, 2),
                "ndvi": ndvi, "temp_max": temp_max, "humidity": humidity,
                "wind_speed": wind_speed, "precip": precip,
                "drought_index": drought_idx,
                "past_hotspot_count": hotspot_cnt, "past_frp": frp,
                "risk_score": risk_score,
                "split": "train" if d.year <= 2021 else ("val" if d.year == 2022 else "test"),
            })

    df = pd.DataFrame(rows).iloc[:min(len(rows), n_cells * 10)]
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(SEASON_LABELS)
    return df


# ---------------------------------------------------------------------------
# Graphique 1 : Saisonnalité mensuelle des feux
# ---------------------------------------------------------------------------

def plot_fire_seasonality(df: pd.DataFrame, out_path: Path):
    """
    Boxplot du nombre de hotspots par mois (toutes années confondues).

    Met en évidence le pic estival des incendies méditerranéens.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hotspots mensuels
    monthly = df.groupby(["date", "month"])["past_hotspot_count"].sum().reset_index()
    ax = axes[0]
    sns.boxplot(data=monthly, x="month", y="past_hotspot_count",
                palette="YlOrRd", ax=ax, linewidth=0.7)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Nombre de hotspots")
    ax.set_title("Saisonnalité mensuelle des feux")
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.grid(True, axis="y", alpha=0.3)

    # FRP mensuel moyen
    ax = axes[1]
    frp_monthly = df[df["past_frp"] > 0].groupby("month")["past_frp"].mean().reset_index()
    ax.bar(frp_monthly["month"], frp_monthly["past_frp"], color="#d7191c", alpha=0.8)
    ax.set_xlabel("Mois")
    ax.set_ylabel("FRP moyen (MW) — cellules actives")
    ax.set_title("Puissance radiative des feux (FRP) par mois")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Saisonnalité des incendies — Méditerranée 2018–2023", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 1 → {out_path}")


# ---------------------------------------------------------------------------
# Graphique 2 : Distribution du NDVI par saison
# ---------------------------------------------------------------------------

def plot_ndvi_distribution(df: pd.DataFrame, out_path: Path):
    """
    Violin plot + KDE de la distribution du NDVI par saison.

    Montre l'assèchement estival de la végétation méditerranéenne.
    """
    if "season" not in df.columns:
        df = df.copy()
        df["month"]  = df["date"].dt.month
        df["season"] = df["month"].map(SEASON_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = ["Hiver", "Printemps", "Été", "Automne"]
    palette = [SEASON_PALETTE[s] for s in order]

    ax = axes[0]
    sns.violinplot(data=df, x="season", y="ndvi", order=order,
                   palette=palette, inner="quartile", ax=ax, linewidth=0.8)
    ax.set_xlabel("Saison")
    ax.set_ylabel("NDVI")
    ax.set_title("Distribution du NDVI par saison")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    for season, color in SEASON_PALETTE.items():
        sub = df[df["season"] == season]["ndvi"].dropna()
        if len(sub) > 0:
            sub.plot.kde(ax=ax, label=season, color=color, linewidth=2)
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Densité")
    ax.set_title("KDE du NDVI par saison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Distribution du NDVI — Bassin méditerranéen", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 2 → {out_path}")


# ---------------------------------------------------------------------------
# Graphique 3 : Matrice de corrélation
# ---------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame, out_path: Path):
    """
    Heatmap de la corrélation de Pearson entre toutes les features et le label.
    """
    cols = FEATURE_COLS + ["risk_score"]
    corr = df[cols].dropna().corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, linewidths=0.5,
        ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Matrice de corrélation des features × label", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 3 → {out_path}")


# ---------------------------------------------------------------------------
# Graphique 4 : Scatter risque vs température
# ---------------------------------------------------------------------------

def plot_risk_vs_temperature(df: pd.DataFrame, out_path: Path):
    """
    Scatter plot risk_score vs temp_max, coloré par saison, avec régression linéaire.
    """
    if "season" not in df.columns:
        df = df.copy()
        df["month"]  = df["date"].dt.month
        df["season"] = df["month"].map(SEASON_LABELS)

    sample = df.sample(min(5000, len(df)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter par saison
    ax = axes[0]
    for season, color in SEASON_PALETTE.items():
        sub = sample[sample["season"] == season]
        ax.scatter(sub["temp_max"], sub["risk_score"],
                   c=color, alpha=0.4, s=8, label=season)
    ax.set_xlabel("Température max (°C)")
    ax.set_ylabel("Risk score")
    ax.set_title("Risque incendie vs Température")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)

    # Risque moyen par décile de température
    ax = axes[1]
    df_c = df.dropna(subset=["temp_max", "risk_score"]).copy()
    df_c["temp_decile"] = pd.qcut(df_c["temp_max"], q=10, labels=False)
    agg = df_c.groupby("temp_decile").agg(
        temp_mean=("temp_max", "mean"),
        risk_mean=("risk_score", "mean"),
        risk_std=("risk_score", "std"),
    ).reset_index()
    ax.errorbar(agg["temp_mean"], agg["risk_mean"], yerr=agg["risk_std"],
                fmt="o-", color="#c0392b", linewidth=2, markersize=6, capsize=4)
    ax.set_xlabel("Température moyenne par décile (°C)")
    ax.set_ylabel("Risque moyen ± std")
    ax.set_title("Risque moyen par décile de température")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Relation Risque–Température — Bassin méditerranéen", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 4 → {out_path}")


# ---------------------------------------------------------------------------
# Graphique 5 : Carte de risque prédit (cartopy ou matplotlib simple)
# ---------------------------------------------------------------------------

def plot_risk_map_mediterranean(
    risk_grid: np.ndarray,
    out_path: Path,
    title: str = "Risque incendie moyen prédit — Méditerranée",
    use_cartopy: bool = True,
):
    """
    Carte du risque incendie moyen sur la Méditerranée.

    Essaie cartopy pour les contours côtiers ; repli sur matplotlib pur.

    Args:
        risk_grid  : array (N_LAT, N_LON) de risques ∈ [0, 1]
        out_path   : chemin de sortie
        title      : titre de la figure
        use_cartopy: tenter d'utiliser cartopy
    """
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fire", ["#2b83ba", "#ffffbf", "#fdae61", "#d7191c"]
    )

    plotted = False
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            fig = plt.figure(figsize=(14, 7))
            ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

            im = ax.pcolormesh(LONS, LATS, risk_grid, cmap=cmap,
                               vmin=0, vmax=1, transform=ccrs.PlateCarree(), shading="auto")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
            ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="gray")
            ax.add_feature(cfeature.LAND,      facecolor="none", edgecolor="none")
            gl = ax.gridlines(draw_labels=True, alpha=0.4, linestyle="--")
            gl.top_labels = gl.right_labels = False
            plt.colorbar(im, ax=ax, label="Risque moyen ∈ [0, 1]", shrink=0.7, pad=0.04)
            ax.set_title(title, fontsize=13, fontweight="bold")
            plotted = True
        except ImportError:
            logger.warning("cartopy non disponible → carte matplotlib simple.")

    if not plotted:
        fig, ax = plt.subplots(figsize=(13, 6))
        im = ax.pcolormesh(LONS, LATS, risk_grid, cmap=cmap, vmin=0, vmax=1, shading="auto")
        plt.colorbar(im, ax=ax, label="Risque moyen ∈ [0, 1]", shrink=0.8)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Carte de risque → {out_path}")


# ---------------------------------------------------------------------------
# Rapport EDA complet
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, figures_dir: Path):
    """
    Lance les 4 graphiques EDA et les sauvegarde dans figures_dir.

    Args:
        df          : DataFrame fusionné (réel ou simulé)
        figures_dir : répertoire de sortie
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    if "month" not in df.columns:
        df = df.copy()
        df["month"]  = df["date"].dt.month
        df["season"] = df["month"].map(SEASON_LABELS)

    plot_fire_seasonality(df, figures_dir / "01_fire_seasonality.png")
    plot_ndvi_distribution(df, figures_dir / "02_ndvi_distribution.png")
    plot_correlation_matrix(df, figures_dir / "03_correlation_matrix.png")
    plot_risk_vs_temperature(df, figures_dir / "04_risk_vs_temperature.png")

    # Carte simulée ou agrégée depuis le dataset
    if "lat_idx" in df.columns and "lon_idx" in df.columns:
        pivot = df.pivot_table(index="lat_idx", columns="lon_idx", values="risk_score", aggfunc="mean")
        risk_grid = pivot.values
    else:
        rng = np.random.default_rng(42)
        summer = np.sin(np.pi * np.linspace(0, 1, N_LAT))[:, None]
        risk_grid = np.clip(0.1 + 0.5 * summer + rng.normal(0, 0.05, (N_LAT, N_LON)), 0, 1).astype(np.float32)

    plot_risk_map_mediterranean(risk_grid, figures_dir / "05_risk_map_eda.png")

    logger.info(f"EDA terminée — {len(list(figures_dir.glob('*.png')))} figures générées dans {figures_dir}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """Lance l'EDA complète en mode simulé."""
    base          = Path(__file__).resolve().parents[1]
    processed_dir = base / "data" / "processed"
    figures_dir   = base / "data" / "figures"

    logger.info("=== EDA (visualize.py) ===")

    df = load_dataset(processed_dir)
    if df is None:
        logger.info("Génération de données simulées …")
        df = simulate_eda_data(n_cells=3000, seed=42)

    run_eda(df, figures_dir)
    print(f"\nFigures sauvegardées dans : {figures_dir}")


if __name__ == "__main__":
    main()
