# Code Guide — Fire Risk Prediction

Ce document explique les concepts clés de chaque fichier source. Le but n'est pas de répéter le code ligne à ligne, mais d'expliquer **pourquoi** chaque décision est prise et **quel problème** elle résout.

---

## Table des matières

1. [pipeline.py — Fusion des données](#1-pipelinepy)
2. [dataset.py — Séquences PyTorch](#2-datasetpy)
3. [model.py — LSTM & TCN](#3-modelpy)
4. [train.py — Boucle d'entraînement](#4-trainpy)
5. [evaluate.py — Métriques & figures](#5-evaluatepy)
6. [visualize.py — EDA](#6-visualizepy)

---

## 1. `pipeline.py`

**Rôle :** transformer trois sources hétérogènes en un seul tableau structuré prêt pour le Deep Learning.

### Le problème de la grille commune

Les trois sources ont des résolutions incompatibles :

| Source | Résolution native | Format |
|--------|------------------|--------|
| MODIS NDVI | 1 km × 1 km, toutes les 16 j | GeoTIFF |
| ERA5 | ~28 km × 28 km, quotidien | NetCDF |
| FIRMS | Points GPS irréguliers | CSV |

La solution est de **projeter tout sur une grille régulière 0.25° × 0.25°** (~28 km), choisie pour correspondre à la résolution native d'ERA5. Les NDVI sont downsampleés (moyenne spatiale via `rasterio.reproject`), les hotspots FIRMS sont "snappés" vers la cellule la plus proche.

```
lat_idx = round( (lat_point - LAT_MIN) / 0.25 )
```

### Pourquoi calculer humidity depuis le point de rosée ?

ERA5 ne fournit pas directement l'humidité relative mais la **température du point de rosée** `d2m`. La formule de **Magnus** est une approximation analytique de la pression de vapeur saturante :

```
RH = 100 × exp(17.625 × Td / (243.04 + Td))
              ─────────────────────────────────
              exp(17.625 × T  / (243.04 + T))
```

Quand `Td` → `T`, l'air est saturé (RH → 100%). C'est plus précis qu'une interpolation linéaire car la courbe de Clausius-Clapeyron est exponentielle.

### Construction du label `risk_score`

C'est la décision la plus importante du pipeline. On n'a pas de "score de risque" dans les données — on le **construit** depuis les feux réels observés dans le futur :

```python
# Pour chaque instant t, le risque est ce qui va se passer dans les 7 jours suivants
future_hotspots = hotspot_series[t+1 : t+8].sum(axis=0)   # somme des feux futurs
future_frp      = frp_series[t+1 : t+8].mean(axis=0)      # intensité moyenne
risk[t] = (future_hotspots + 0.01 × future_frp) / max_global
```

**Pourquoi 7 jours ?** Les modèles météo sont utiles à cet horizon. Au-delà, le signal dépend trop de la variabilité chaotique de l'atmosphère.

**Pourquoi le FRP pondéré à 0.01 ?** Le FRP (Fire Radiative Power, en MW) est en général 100× plus grand numériquement que le comptage de hotspots. Le coefficient 0.01 le ramène à la même échelle sans masquer l'information spatiale du comptage.

### L'indice de sécheresse

```python
drought_index[t] = Σ temp[t-30:t] / (Σ precip[t-30:t] + 1)
```

C'est une approximation de l'**Evapotranspiration Potential** sur 30 jours : plus il fait chaud et moins il pleut, plus le sol est sec. Le `+1` au dénominateur évite la division par zéro les jours sans pluie.

### Split temporel strict

```
Train : 2018–2021   Val : 2022   Test : 2023
```

On ne coupe **jamais** aléatoirement sur des séries temporelles — cela créerait une **fuite d'information** (data leakage) car des jours consécutifs sont très corrélés. Le split chronologique garantit que le modèle ne "voit" jamais le futur.

---

## 2. `dataset.py`

**Rôle :** convertir le tableau 2D (temps × cellules) en séquences exploitables par un LSTM.

### Pourquoi des séquences glissantes ?

Un LSTM a besoin de **contexte temporel** pour détecter des patterns comme "3 jours chauds après 2 semaines sans pluie → feu probable". La fenêtre glissante de 15 jours (`SEQ_LEN=15`) :

```
Jour t-14, t-13, …, t-1, t  →  prédit risk_score[t+1]
```

Pour une grille de `N_lat × N_lon` cellules et `T` jours, on obtient `N_cells × (T - SEQ_LEN + 1)` séquences indépendantes. Chaque cellule géographique est traitée comme un individu distinct.

### MinMaxScaler : pourquoi le fitter sur train uniquement ?

```python
if is_train:
    self.scaler = FeatureScaler()
    X_norm = self.scaler.fit_transform(X_train)   # apprend min/max ICI
else:
    X_norm = scaler.transform(X_val_or_test)       # applique les mêmes min/max
```

Si on fittait sur val ou test, le modèle aurait une information implicite sur la distribution future → c'est du **data leakage**. Le scaler "voit" les extremes de 2018–2021 et normalise 2022–2023 avec ces mêmes bornes.

### Weighted Sampler : gérer le déséquilibre

Les feux sont rares : moins de 5% des cellules/jours ont un `risk_score > 0.5`. Sans correction, le modèle apprendrait à prédire 0 partout (MAE correcte, mais inutile).

```python
weights[risk > 0.5] = 10.0   # ces exemples sont tirés 10× plus souvent
weights[risk ≤ 0.5] = 1.0
sampler = WeightedRandomSampler(weights, replacement=True)
```

`replacement=True` est obligatoire : on "sur-échantillonne" les exemples rares sans dupliquer physiquement les données.

---

## 3. `model.py`

**Rôle :** deux architectures qui transforment une séquence `(15, 8)` en un scalaire ∈ [0, 1].

### LSTM — pourquoi ce choix pour des séries temporelles météo ?

Le **Long Short-Term Memory** résout le problème du **gradient qui disparaît** dans les RNN simples grâce à ses portes :

```
Forget gate  : "oublie" les informations devenues inutiles
Input gate   : "retient" les nouvelles informations pertinentes
Output gate  : "lit" la mémoire pour produire la sortie
```

Pour le risque incendie, cela permet de mémoriser "pas de pluie depuis 10 jours" même si les derniers jours ne montrent aucun signe direct.

```
Architecture :
  Input (15, 8)
    → LSTM layer 1  (hidden=64)  +  dropout 20%
    → LSTM layer 2  (hidden=64)
    → dernier état caché h[-1]   (64,)
    → Linear(64→64) + ReLU + Dropout
    → Linear(64→1) + Sigmoid
  Output : scalaire ∈ [0, 1]
```

La **Sigmoid finale** est essentielle : elle contraint la sortie dans [0, 1] ce qui est cohérent avec la définition du `risk_score`.

### TCN — pourquoi une baseline convolutive ?

Le **Temporal CNN** remplace la récurrence par des **convolutions dilatées** :

```
Dilation 1  : vue sur 3 jours consécutifs
Dilation 2  : vue sur 5 jours (saute 1 jour entre chaque)
Dilation 4  : vue sur 9 jours
Dilation 8  : vue sur 17 jours
```

L'empilement exponentiel couvre `2^n × (kernel-1)` pas de temps avec seulement `n` couches — très efficace. Le TCN est souvent **plus rapide à entraîner** que le LSTM (parallélisable) mais capte moins bien les dépendances à très long terme.

La **connexion résiduelle** dans chaque `TemporalBlock` stabilise le gradient :
```
output = ReLU( conv(x) + skip(x) )
```
Si les convolutions n'apportent rien, le gradient passe directement par `skip` → pas de dégradation.

### Initialisation orthogonale des poids LSTM

```python
nn.init.orthogonal_(weight_hh)   # matrices de transition
nn.init.xavier_uniform_(weight_ih)   # matrices d'entrée
```

L'initialisation orthogonale préserve la norme du gradient dans les matrices de transition `h → h`. C'est particulièrement important pour les LSTM sur des séquences longues où le gradient peut exploser ou s'effondrer dès les premières époques.

---

## 4. `train.py`

**Rôle :** orchestrer l'optimisation du modèle avec toutes les bonnes pratiques.

### Weighted MSE — pourquoi ne pas utiliser MSE standard ?

Avec un déséquilibre fort (5% de cellules à risque élevé), la MSE standard est minimisée en prédisant ~0 partout :

```
MSE standard : erreur moyenne sur 95% de cas faciles + 5% de cas difficiles
Weighted MSE : les 5% de cas difficiles comptent 10× plus
```

```python
weights = where(target > 0.5,  10.0,  1.0)
loss = mean( weights × (pred - target)² )
```

Le poids 10 a été choisi empiriquement pour équilibrer l'importance des deux classes sans déstabiliser l'entraînement.

### Gradient clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Si le gradient devient trop grand (explosion), les poids peuvent partir à `±∞` en une seule étape. Le clipping plafonne la **norme globale** du gradient à 1.0, ce qui stabilise l'entraînement sur les longues séquences LSTM.

### ReduceLROnPlateau — scheduler adaptatif

```
Si val_loss ne diminue pas pendant 5 époques → lr × 0.5
```

Au lieu de diminuer lr à intervalles fixes ("step decay"), ce scheduler réagit à la **dynamique réelle** de l'entraînement. C'est particulièrement utile ici car la convergence dépend du contenu du batch (rare vs commun).

### Early stopping — patience=10

```python
if val_loss n'améliore pas pendant 10 epochs → arrêt
```

Sans early stopping, le modèle finirait par mémoriser les patterns du train set (overfitting). La patience de 10 laisse suffisamment de marge pour que le scheduler puisse réduire le lr et relancer une amélioration avant de s'arrêter.

### Checkpoint du meilleur modèle

```python
torch.save({"epoch": epoch, "model_state": model.state_dict(), ...}, best_model.pt)
```

On sauvegarde l'état complet (poids + optimiseur + métriques) à chaque fois que `val_loss` s'améliore. À la fin, le meilleur modèle n'est pas forcément le dernier — c'est celui du checkpoint.

---

## 5. `evaluate.py`

**Rôle :** mesurer la qualité réelle du modèle avec quatre angles complémentaires.

### Pourquoi quatre métriques différentes ?

| Métrique | Ce qu'elle mesure | Limite |
|----------|------------------|--------|
| **MAE** | Erreur absolue moyenne en points de risque | Sensible aux unités, insensible aux rangs |
| **RMSE** | Pénalise davantage les grandes erreurs (carré) | Sensible aux outliers |
| **Spearman** | Cohérence des rangs (cellule A plus risquée que B ?) | Ne mesure pas l'amplitude |
| **AUC-ROC** | Capacité à séparer risque élevé vs faible | Dépend du seuil de binarisation |

Un bon modèle a les quatre : MAE/RMSE basses ET Spearman/AUC hautes. Un modèle qui prédit bien les rangs mais mal les valeurs absolues aura un bon Spearman mais un MAE élevé.

### Spearman plutôt que Pearson

La corrélation de Pearson mesure la linéarité. La corrélation de **Spearman** (sur les rangs) mesure la **monotonie** : "est-ce que les cellules prédites à haut risque ont bien tendance à avoir un vrai risque élevé ?" C'est plus robuste aux outliers et approprié pour une variable ∈ [0, 1] avec distribution asymétrique.

### AUC-ROC avec binarisation à 0.5

```python
y_bin = (y_true >= 0.5).astype(int)   # "feu probable" vs "pas de feu"
auc = roc_auc_score(y_bin, y_pred)    # y_pred reste continu
```

L'AUC-ROC mesure la probabilité que le modèle classe correctement une paire (cellule risquée, cellule sûre). Une AUC=0.9 signifie que dans 90% des cas, la cellule vraiment risquée reçoit un score prédit plus élevé. Elle est indépendante du seuil de décision (contrairement à l'accuracy).

---

## 6. `visualize.py`

**Rôle :** comprendre les données avant de modéliser (EDA) et présenter les résultats.

### Figure 1 — Saisonnalité (boxplot mensuel)

Montre que les feux méditerranéens suivent un cycle annuel très marqué avec un pic en juillet–août. C'est la validation visuelle que nos données (réelles ou simulées) reproduisent ce comportement connu.

### Figure 2 — Distribution NDVI par saison

Le NDVI (Normalized Difference Vegetation Index) est le rapport `(NIR - Red) / (NIR + Red)`. En été méditerranéen, la végétation se dessèche → NDVI chute. La figure valide que cette anti-corrélation saisonnière est présente dans les données.

### Figure 3 — Matrice de corrélation

Révèle les redondances et les prédicteurs clés. En général :
- `temp_max` ↑ corrèle fortement avec `drought_index` ↑
- `humidity` ↑ corrèle avec `risk_score` ↓ (humidité protège)
- `ndvi` ↑ corrèle faiblement avec `risk_score` ↓ (végétation verte moins inflammable)

C'est un test de cohérence : si `humidity` corrèle positivement avec le risque, quelque chose est inversé.

### Figure 4 — Scatter risque vs température

La relation n'est pas linéaire mais sigmoïdale : en dessous de ~15°C, le risque reste faible quelle que soit la température ; au-dessus de ~30°C, le risque sature. La visualisation par déciles confirme cette non-linéarité que le LSTM doit capturer.

### Figure 5 — Carte de risque

Carte en couleur bleu→jaune→rouge (`LinearSegmentedColormap`) sur la grille 0.25°. Elle permet de localiser géographiquement les zones à risque. En Méditerranée, les zones typiquement à haut risque sont le nord de l'Espagne, la Grèce et la Turquie en été.

---

## Flux de données de bout en bout

```
data/raw/
  firms/*.csv    →  load_firms_csv()    →  hotspot_count, frp_mean   ┐
  ndvi/*.tif     →  load_ndvi_geotiff() →  ndvi (N_lat, N_lon)       ├→ run_pipeline()
  era5/*.nc      →  load_era5_netcdf()  →  temp, humidity, wind, …   ┘
                                                  │
                                    build_risk_score()   ← label supervisé
                                    build_drought_index() ← feature engineered
                                                  │
                              data/processed/
                                X_train.npy  (T, N_lat, N_lon, 8)
                                y_train.npy  (T, N_lat, N_lon)
                                                  │
                              FireRiskDataset.__getitem__()
                                x : (15, 8)   ← fenêtre glissante
                                y : scalar    ← risk_score cible
                                                  │
                              FireRiskLSTM.forward()
                                LSTM(15,8) → h[-1](64,) → Linear → Sigmoid → ŷ ∈ [0,1]
                                                  │
                              weighted_mse_loss(ŷ, y)
                              Adam + ReduceLROnPlateau + EarlyStopping
                                                  │
                              evaluate_model()
                                MAE | RMSE | Spearman | AUC
```

---

## Paramètres clés et justification

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `SEQ_LEN` | 15 jours | Capture les tendances de 2 semaines (sécheresse courte) |
| `FORECAST_HORIZON` | 7 jours | Horizon utile pour la prévention incendie |
| `HIDDEN_SIZE` | 64 | Compromis capacité/temps CPU (pas de GPU requis) |
| `HIGH_RISK_WEIGHT` | 10× | Équilibre ~5% positifs vs ~95% négatifs |
| `PATIENCE` | 10 epochs | Laisse le scheduler agir avant d'arrêter |
| `LR` | 1e-3 | Valeur par défaut Adam, bien testée pour LSTM |
| `SEED` | 42 | Reproductibilité totale sur tous les composants |
