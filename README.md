# Where will Germany's next EV charging stations be built?

Data science project from a course during my semester abroad (Oct 2025,
revised Oct 2026) on the public
charging station register of the German Federal Network Agency
(Bundesnetzagentur, report of 24 Sep 2025, ~95,500 operational stations).

**Question:** Given today's charging infrastructure, which areas in Germany are
most likely to get new charging stations in the next two years?

## Approach

1. **Exploration and cleaning** (`exploration_and_data_science.ipynb`):
   descriptive analysis of the register (power, operators, states, growth over
   time, a hypothesis test on station count vs. average power, map) and export
   of a reduced CSV for step 2.
2. **Location model** (`Charging Station_Location_Prediction.ipynb`):
   - Germany's bounding box is split into a grid of 0.05° cells (~5 km,
     34,000 cells). Only cells with at least one pre-2024 station within 0.5°
     are kept (26,088 cells), a rough approximation of "on land, in or close
     to Germany".
   - **Temporal setup:** all features are computed from stations commissioned
     **before 2024**. Target: the cell gets **at least one new station in
     2024-2025** (2025 is partial, until 24 Sep). 39% of cells are positive;
     among cells without any station before 2024 it is 7%.
   - 15 features per cell: whether the cell already has a station, distance
     to the nearest station, station counts within 0.1°/0.2°/0.5°, density
     gradient, distance to the ten largest cities, position, recent vs. older
     stations nearby, local growth trend, average charging power nearby.
     Neighbourhood queries use `cKDTree.query_ball_point`.
   - Random Forest, Gradient Boosting and XGBoost, class imbalance handled with
     class/sample weights (no downsampling).
   - **Evaluation:** spatial block cross-validation (`GroupKFold`, 5 folds,
     1° x 1° blocks, 85 blocks). The blocks (~111 x 70 km) are larger than most
     feature radii, so test areas are not surrounded by training cells.
   - **Forecast:** the best model (by spatial PR-AUC) is applied to features
     computed from all stations up to 2025. The score ranks cells by how
     likely a new station is within roughly the next two years, assuming the
     2024-2025 pattern continues. It is a ranking, not a calibrated
     probability for 2026. The top 500 cells without a station are exported to
     `outputs/predicted_new_locations.csv`.

## Results

Spatial block CV (5 folds, mean ± std, threshold 0.5 for F1/precision/recall).
The baseline scores cells by `stations_within_10km` and predicts "new station"
for every cell that already has one.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Baseline (density only) | 0.872 ± 0.019 | 0.796 ± 0.060 | 0.765 ± 0.037 | 0.656 ± 0.042 | 0.917 ± 0.024 |
| Random Forest | 0.893 ± 0.014 | 0.829 ± 0.047 | 0.771 ± 0.036 | 0.702 ± 0.031 | 0.857 ± 0.049 |
| Gradient Boosting | 0.888 ± 0.015 | 0.823 ± 0.048 | 0.757 ± 0.041 | 0.702 ± 0.034 | 0.823 ± 0.061 |
| XGBoost | 0.893 ± 0.015 | 0.829 ± 0.048 | 0.768 ± 0.038 | 0.703 ± 0.032 | 0.849 ± 0.055 |

Only cells **without** a station before 2024 (new locations, 7% positive,
out-of-fold spatial predictions):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Baseline (density only) | 0.817 | 0.213 |
| Random Forest | 0.844 | 0.216 |
| Gradient Boosting | 0.842 | 0.219 |
| XGBoost | 0.851 | 0.227 |

Takeaways:

- The models beat the density baseline, but only by about 0.02 ROC-AUC and
  0.03 PR-AUC. Most of the signal is "new stations appear where stations
  already are" (`has_station_before` and `dist_to_nearest_station` dominate
  the feature importance).
- Predicting genuinely new locations is much harder: PR-AUC ~0.22 at a 7%
  base rate.

## What changed and why

The first version (Oct 2025) reported ROC-AUC 0.97-0.98:

| Model (old setup) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.930 | 0.814 | 0.933 | 0.869 | 0.983 |
| Gradient Boosting | 0.923 | 0.842 | 0.851 | 0.846 | 0.976 |
| XGBoost | 0.899 | 0.725 | 0.958 | 0.825 | 0.974 |

These numbers were not meaningful, for two reasons:

1. **Target leakage through density features.** The target ("is there a
   station in this cell?") and features such as `stations_within_10km` were
   computed from the same set of stations. A cell with a station
   automatically has at least one station within 10 km, so the model largely
   recovered the target from its own inputs. Fix: features only from stations
   before 2024, target = new stations in 2024-2025.
2. **Random train/test split on spatial data.** Neighbouring cells share
   almost identical features and labels, so a random split puts near-copies of
   test cells into the training set. In addition, the empty cells had been
   downsampled before the split, so the test set did not reflect the real class
   ratio. Fix: spatial block CV and class weights instead of downsampling.

With the leakage fixed, the random split still inflates the scores, though
only moderately (same pipeline, 5 folds):

| Model | ROC-AUC random | ROC-AUC spatial | PR-AUC random | PR-AUC spatial |
|---|---|---|---|---|
| Random Forest | 0.904 ± 0.004 | 0.893 ± 0.014 | 0.850 ± 0.006 | 0.829 ± 0.047 |
| Gradient Boosting | 0.903 ± 0.004 | 0.888 ± 0.015 | 0.848 ± 0.006 | 0.823 ± 0.048 |
| XGBoost | 0.905 ± 0.004 | 0.893 ± 0.015 | 0.851 ± 0.006 | 0.829 ± 0.048 |

The random split also hides the variation between regions (std 0.004 vs.
0.014-0.015 ROC-AUC). Most of the drop from 0.98 to 0.89 comes from removing
the leakage and changing the target, not from the split.

Other changes: vectorised neighbourhood features (the `iterrows` loops are
gone), relative `data/` paths, cells far outside Germany / at sea excluded,
`requirements.txt` turned into a package list.

## Limitations

- **Grid in degrees, not kilometres.** 0.05° is ~5.6 km north-south but only
  ~3.2-3.8 km east-west, so "10 km" radii vary with latitude.
- **Only station data as input.** Terrain, road access, parking, land use and
  grid connection are missing, so the model cannot tell whether a charging
  station can be built at a location at all. Population and traffic are
  missing as well. The model mainly extrapolates the existing station pattern.
- **Study area is approximate.** Keeping cells within 0.5° of an existing
  station still includes some foreign cells near the border.
- **Remaining spatial dependence.** There is no buffer zone between CV blocks,
  and the 0.3-0.5° features reach across block borders, so the spatial scores
  are still slightly optimistic.
- **Forecast.** 2025 is a partial year, and the features for the forecast
  (all stations up to 2025) are shifted compared with the training snapshot
  (before 2024). The output is a ranking under the assumption that recent
  dynamics continue.

## Run it

1. Download the raw register CSV (too large for GitHub):
   [Google Drive](https://drive.google.com/file/d/1SliuVz9lS0_ynGpnjPcKmmdJo-H-gP6k/view?usp=sharing)
   and save it as `data/Ladesaeulenregister_BNetzA.CSV` (`data/` is
   git-ignored).
2. Python 3.11 or 3.12, then `pip install -r requirements.txt`.
3. Run `exploration_and_data_science.ipynb` first. It writes
   `data/Ladesaeulenregister_BNetzA_smaller.CSV`.
4. Run `Charging Station_Location_Prediction.ipynb` (about 3 minutes on a
   laptop CPU). It writes `outputs/predicted_new_locations.csv`.

Data source: [Bundesnetzagentur – E-Mobilität](https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/start.html)
