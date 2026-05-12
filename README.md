# Fire Risk Prediction — Bassin méditerranéen

Prédiction du risque d'incendie sur le bassin méditerranéen par Deep Learning (LSTM / TCN).

## Zone & données

| Paramètre | Valeur |
|-----------|--------|
| Zone | 35°N–47°N, 5°W–28°E |
| Grille | 0.25° × 0.25° (49 × 133 cellules) |
| Période | 2018–2023 |
| Tâche | Régression — risk score ∈ [0, 1] |
| Horizon | 7 jours futurs |

### Sources de données

| Source | Format | Variables |
|--------|--------|-----------|
| NASA FIRMS | CSV | latitude, longitude, acq_date, frp |
| MODIS NDVI | GeoTIFF 1 km 16j | NDVI ∈ [-1, 1] |
| ERA5 | NetCDF journalier | t2m, d2m, u10, v10, tp |

## Structure du projet

```
fire_risk_project/
├── data/
│   ├── raw/
│   │   ├── firms/          # CSV NASA FIRMS
│   │   ├── ndvi/           # GeoTIFF MODIS
│   │   └── era5/           # NetCDF ERA5
│   ├── processed/          # Parquet, .npy, checkpoints, scaler
│   └── figures/            # PNG générés
├── src/
│   ├── pipeline.py         # Fusion des 3 sources → Parquet + npy
│   ├── dataset.py          # PyTorch Dataset (séquences 15j)
│   ├── model.py            # LSTM + TCN baseline
│   ├── train.py            # Boucle entraînement + early stopping
│   ├── evaluate.py         # MAE, RMSE, Spearman, AUC, cartes
│   └── visualize.py        # EDA (4 figures) + carte de risque
├── notebooks/
│   └── exploration.ipynb   # Démonstration interactive
├── requirements.txt
└── README.md
```

## Installation

```bash
# Créer l'environnement virtuel (déjà présent dans venv/)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/macOS

pip install -r requirements.txt
```

## Utilisation

### Mode simulation (sans données réelles)

Tous les modules détectent automatiquement l'absence de données et basculent en mode simulé.

```bash
# 1. Pipeline (simulation)
python src/pipeline.py

# 2. EDA
python src/visualize.py

# 3. Entraînement LSTM + TCN
python src/train.py

# 4. Évaluation
python src/evaluate.py
```

### Avec des données réelles

1. Placer les CSV FIRMS dans `data/raw/firms/`
2. Placer les GeoTIFF NDVI dans `data/raw/ndvi/`
3. Les fichiers ERA5 sont dans `data/raw/era5/` (télécharger avec `download_era5.py`)
4. Lancer `python src/pipeline.py` — le pipeline détecte automatiquement les fichiers

## Features du modèle

| Feature | Description |
|---------|-------------|
| `ndvi` | Indice de végétation normalisé |
| `temp_max` | Température maximale (°C) |
| `humidity` | Humidité relative (%, formule Magnus) |
| `wind_speed` | Vitesse du vent (m/s = √u²+v²) |
| `precip` | Précipitations (mm) |
| `drought_index` | Sécheresse cumulée 30j |
| `past_hotspot_count` | Nb hotspots récents |
| `past_frp` | FRP moyen récent (MW) |

## Splits temporels

| Split | Période |
|-------|---------|
| Train | 2018–2021 |
| Val | 2022 |
| Test | 2023 |

## Métriques de référence (simulées)

| Modèle | MAE | RMSE | Spearman | AUC |
|--------|-----|------|----------|-----|
| LSTM   | ~0.08 | ~0.11 | ~0.82 | ~0.91 |
| TCN    | ~0.09 | ~0.12 | ~0.79 | ~0.89 |

## Reproductibilité

Toutes les graines aléatoires sont fixées à `42`. Lancer `train.set_seed(42)` avant tout entraînement.
