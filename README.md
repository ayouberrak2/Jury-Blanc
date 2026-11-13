# Jury Blanc

1. `notebooks/01_eda.ipynb` : exploration et preparation.
2. `notebooks/02_modelisation.ipynb` : comparaison initiale des modeles.
3. `notebooks/03_optimisation.ipynb` : validation croisee, optimisation et scores finaux.

`metrique/` contient seulement :
- `validation_test_metrics.csv` : metriques train, validation et test.
- `cv_fold_metrics.csv` : progression des scores par pli.
- `accuracy_cv.png` : accuracy des plis et point test final.
- `accuracy_train_validation_test.png` : comparaison globale.

Les exports detailles precedents sont archives dans `reports/archive_metrique/`.
Les notebooks 04 a 06 sont des essais supplementaires ; leurs nouveaux exports vont dans `reports/`.
Le notebook 06 utilise la comparaison produite par le notebook 05.
Les pipelines sauvegardes sont dans `models/`.

Installer : `python -m pip install -r requirements.txt`.
Lancer : `python -m jupyter notebook`.
Le test a deja ete consulte ; ne pas l'utiliser pour de nouveaux reglages.
Projet pedagogique, sans usage clinique.


## Feature engineering

`notebooks/07_feature_engineering.ipynb` compare trois variables seules et ensemble.
`src/features.py` calcule les variables apres imputation, avant standardisation.
L'option `feature_set` de `build_preprocessor` vaut `none` par defaut.
Resultats : `reports/feature_engineering/comparison.csv` et `folds.csv`.
Aucun gain observe ; la reference sans nouvelles variables est conservee.
Le candidat experimental est dans `models/feature_engineering_candidate.joblib`.
Les quatre fichiers de metrique et le modele retenu auparavant ne sont pas modifies.
