# FireRisk — Prédiction du risque d'incendie (Bassin méditerranéen)

> **Module :** Deep Learning &nbsp;|&nbsp; **Filière :** DATA &nbsp;|&nbsp; **École :** INPT  
> **Auteurs :** Ilyass Aamoum &amp; Chihab Chouai &nbsp;|&nbsp; **Encadrant :** Mr. Tarik Fissaa

---

## Vue d'ensemble

FireRisk est un système de prédiction spatiale du risque d'incendie sur le bassin méditerranéen.
Il fusionne trois sources de données satellitaires et de réanalyse climatique (ERA5, MODIS NDVI, NASA FIRMS),
construit des séquences temporelles de 15 jours par cellule de grille 0,25°, et entraîne deux
architectures de Deep Learning (LSTM et TCN) pour estimer, à chaque point de la grille, un score de
risque normalisé ∈ [0, 1] représentant l'intensité des foyers actifs attendus dans les 7 jours suivants.

Le projet inclut une interface web interactive construite avec Streamlit (thème sombre, couleurs feu)
et un rapport technique complet de 45 pages rédigé en LaTeX.

---

## Zone d'étude et données

| Paramètre | Valeur |
|-----------|--------|
| Zone géographique | 35°N – 47°N, 5°W – 28°E |
| Résolution de la grille | 0,25° × 0,25° (49 × 133 = 6 517 cellules) |
| Période d'entraînement | 2022-01-01 → 2022-12-15 |
| Période de validation | 2022-12-16 → 2022-12-31 |
| Période de test | 2023-01-01 → 2023-12-31 |
| Tâche | Régression — risk score ∈ [0, 1] |
| Horizon de prédiction | 7 jours |

### Sources de données

| Source | Format | Variables utilisées |
|--------|--------|---------------------|
| **ERA5** (ECMWF) | NetCDF journalier | `t2m`, `d2m`, `u10`, `v10`, `tp` |
| **MODIS NDVI** (Terra MOD13A2) | GeoTIFF 1 km / 16 jours | NDVI (facteur ×0,0001) |
| **NASA FIRMS** (MODIS/VIIRS) | CSV journalier | `latitude`, `longitude`, `acq_date`, `frp` |

---

## Features du modèle

| Feature | Description |
|---------|-------------|
| `ndvi` | Indice de végétation normalisé (MODIS, reprojection 0,25°) |
| `temp_max` | Température maximale journalière (°C, ERA5 t2m) |
| `humidity` | Humidité relative (%, formule Magnus-Tetens depuis t2m et d2m) |
| `wind_speed` | Vitesse du vent (m/s, √u10²+v10²) |
| `precip` | Précipitations journalières cumulées (mm, ERA5 tp) |
| `drought_index` | Déficit hydrique cumulé sur 30 jours (précip – évapotranspiration) |
| `past_hotspot_count` | Nombre de foyers FIRMS détectés dans la cellule sur 7 jours passés |
| `past_frp` | Fire Radiative Power moyen des foyers passés (MW) |

**Variable cible :** `risk_score` = convolution spatiale (noyau 3×3) des foyers des 7 jours futurs,
normalisée par le 99e percentile du jeu d'entraînement.

---

## Architectures

### LSTM — `FireRiskLSTM`

- 2 couches LSTM empilées, `hidden_size = 64`, `dropout = 0,2`
- Initialisation orthogonale des poids récurrents
- Couche linéaire finale + sigmoid → score ∈ [0, 1]

### TCN — `FireRiskTCN`

- 4 blocs convolutifs causaux dilatés (dilations : 1, 2, 4, 8), `kernel_size = 3`
- Champ réceptif effectif : **61 jours**
- Global average pooling + projection linéaire + sigmoid

### Entraînement commun

- **Loss :** MSE pondérée — poids ×10 pour les exemples à `risk_score > 0,5` (déséquilibre de classe)
- **Sampler :** `WeightedRandomSampler` pour sur-échantillonner les zones à haut risque
- **Optimiseur :** Adam (lr = 1 × 10⁻³, weight decay = 1 × 10⁻⁵)
- **Early stopping :** patience = 5 épochs sur la validation loss
- **Scaler :** Min-Max, ajusté uniquement sur le jeu d'entraînement (`data/processed/scaler.pkl`)

---

## Résultats

| Modèle | MAE | RMSE | Spearman | AUC-ROC |
|--------|-----|------|----------|---------|
| **LSTM** | 0.0885 | 0.1076 | 0.8486 | 0.9142 |
| **TCN** | 0.1196 | 0.1512 | 0.8166 | 0.9017 |

Le LSTM surpasse le TCN sur toutes les métriques, ce qui suggère que les dépendances temporelles
longues (mémoire de l'humidité et des foyers passés) sont mieux capturées par les cellules récurrentes
que par les convolutions dilatées sur cet horizon de 15 jours.

---

## Structure du projet

```
fire_risk_project/
├── app.py                      # Interface Streamlit (3 onglets, thème sombre)
├── requirements.txt            # Dépendances pipeline + entraînement
├── requirements_app.txt        # Dépendances interface Streamlit
├── download_era5.py            # Téléchargement des données ERA5 via CDS API
├── download_2023.py            # Téléchargement complémentaire 2023
│
├── src/
│   ├── pipeline.py             # Fusion ERA5 + NDVI + FIRMS → Parquet + .npy
│   ├── dataset.py              # PyTorch Dataset (fenêtres glissantes 15j)
│   ├── model.py                # FireRiskLSTM + FireRiskTCN
│   ├── train.py                # Boucle d'entraînement, EarlyStopping, weighted MSE
│   ├── evaluate.py             # MAE, RMSE, Spearman, AUC, cartes de risque
│   ├── inference.py            # Prédiction ponctuelle et grille complète
│   └── visualize.py            # EDA (saisonnalité, NDVI, corrélations)
│
├── data/
│   ├── raw/
│   │   ├── era5/               # Fichiers NetCDF ERA5 2022
│   │   ├── era5_2023/          # Fichiers NetCDF ERA5 2023
│   │   ├── firms/              # CSV NASA FIRMS 2022
│   │   ├── firms_2023/         # CSV NASA FIRMS 2023
│   │   └── ndvi/               # GeoTIFF MODIS NDVI
│   └── processed/
│       ├── dataset_2022.parquet
│       ├── dataset_2023.parquet
│       ├── X_train.npy / y_train.npy
│       ├── X_val.npy   / y_val.npy
│       ├── X_test.npy  / y_test.npy
│       ├── scaler.pkl
│       ├── best_model_lstm.pt
│       ├── best_model_tcn.pt
│       ├── history_lstm.json
│       └── history_tcn.json
│
├── notebooks/
│   └── exploration.ipynb       # Analyse exploratoire interactive
│
└── report/
    ├── fire_risk_report.tex    # Rapport LaTeX (45 pages)
    ├── fire_risk_report.pdf    # Rapport compilé
    └── compile.ps1             # Script de compilation (Windows / MiKTeX)
```

---

## Installation

### Prérequis

- Python 3.10+
- CUDA 11.8+ (optionnel, pour l'accélération GPU)

### Environnement

```bash
# Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

# Installer les dépendances
pip install -r requirements.txt
```

Pour l'interface Streamlit uniquement :

```bash
pip install -r requirements_app.txt
```

---

## Utilisation

### 1. Pipeline de données

```bash
# Fusion ERA5 + NDVI + FIRMS → fichiers Parquet et tableaux NumPy
python src/pipeline.py
```

### 2. Entraînement

```bash
# Entraîne LSTM et TCN, sauvegarde les checkpoints et l'historique
python src/train.py
```

### 3. Évaluation

```bash
# Calcule MAE, RMSE, Spearman, AUC ; génère les cartes de risque
python src/evaluate.py
```

### 4. Interface Streamlit

```bash
streamlit run app.py
```

L'interface s'ouvre automatiquement sur `http://localhost:8501` et propose trois onglets :

| Onglet | Contenu |
|--------|---------|
| **Prédiction** | Sélection d'une localisation, score de risque en temps réel, séries temporelles des features, importance des variables |
| **Carte de risque** | Carte de densité interactive (fond sombre), foyers FIRMS superposés, statistiques grille |
| **Évaluation** | Métriques LSTM / TCN, courbes d'apprentissage, nuage de points prédit vs réel |

---

## Rapport technique

Le rapport complet (45 pages, LaTeX) se trouve dans `report/fire_risk_report.pdf`.

Pour recompiler depuis les sources :

```powershell
# Windows — nécessite MiKTeX ou TeX Live
.\report\compile.ps1
```

Le rapport couvre : zone d'étude, sources de données, ingénierie des features, architectures LSTM/TCN,
protocole d'entraînement, évaluation, pipeline logiciel, interface Streamlit, discussion et perspectives.

---

## Reproductibilité

Toutes les graines aléatoires sont fixées à `42` (Python `random`, NumPy, PyTorch).
Les artefacts d'entraînement (scalers, checkpoints, historiques) présents dans `data/processed/`
permettent de relancer directement l'évaluation et l'interface sans ré-entraîner les modèles.

---

## Dépendances principales

| Package | Version minimale | Rôle |
|---------|-----------------|------|
| `torch` | 2.1.0 | Modèles LSTM / TCN |
| `streamlit` | 1.32.0 | Interface web |
| `plotly` | 5.18.0 | Visualisations interactives |
| `xarray` | 2023.1.0 | Lecture NetCDF ERA5 |
| `rasterio` | 1.3.0 | Lecture / reprojection GeoTIFF NDVI |
| `pandas` | 2.0.0 | Traitement des données tabulaires |
| `scikit-learn` | 1.3.0 | MinMaxScaler, métriques |
| `scipy` | 1.11.0 | Corrélation de Spearman |

---

## Licence

Ce projet est réalisé dans un cadre académique (INPT — Filière DATA, Module Deep Learning).
Toute réutilisation doit mentionner les auteurs et l'établissement d'origine.
