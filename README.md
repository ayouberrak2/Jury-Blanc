# Jury Blanc - classification du bien-etre etudiant

Prototype pedagogique : aucun diagnostic, depistage clinique, usage de selection ou de sanction.

## Donnees et execution

Installer les dependances : `python -m pip install -r requirements.txt`.
Placer le fichier dans `data/raw/student_mental_health.csv` :
[dataset Kaggle](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students).
Les donnees brutes, les modeles et les rapports ne sont pas versionnes.
Lancer Jupyter avec `python -m jupyter notebook`, puis executer les cellules dans l'ordre :

1. `notebooks/01_eda.ipynb` : audit, statistiques, relations et preparation.
2. `notebooks/02_modelisation.ipynb` : comparaison initiale sur validation simple.
3. `notebooks/03_optimisation.ipynb` : validation imbriquee, optimisation et bilan final.

Le guide detaille est dans `ss.txt`.

## Modele et metriques

Seul `models/selected_model.joblib` est sauvegarde apres selection par F1 de validation.
Il contient le pipeline complet, `target_encoder`, les classes et les parametres.
La cible utilise OrdinalEncoder avec l'ordre Low=0, Medium=1, High=2.
Cet ordre ne transforme pas les classifieurs en methodes de classification ordinale.
Le test a deja ete consulte : ne pas l'utiliser pour de nouveaux reglages.

`metrique/` contient uniquement :
- `validation_test_metrics.csv`
- `cv_fold_metrics.csv`
- `accuracy_cv.png`
- `accuracy_train_validation_test.png`

## Application Streamlit

Apres creation du modele par le notebook 03 :

```powershell
docker compose up --build -d streamlit
```

Ouvrir http://localhost:8501.
L'image `juryblanc-streamlit:local` est definie par `Dockerfile.streamlit`.
Le modele, les donnees et les metriques sont montes en lecture seule.
L'application affiche le bilan du modele charge et permet de simuler un profil fictif.
Elle n'entraine pas de modele. Les comparaisons du notebook sont presentees comme historiques.

Pour un lancement sans Docker : `python -m streamlit run app/streamlit_app.py`.
Le theme sombre est dans `.streamlit/config.toml`, les styles dans `app/style.css`.
Apres modification de l'interface, reconstruire le conteneur avec la commande ci-dessus.
Arreter : `docker compose stop streamlit`.

## Fonctions pour le futur DAG Airflow

`src/airflow.py` expose `prepare_data`, `preprocess_data`, `train_selected_model`
et `evaluate_and_save`. Chaque execution doit fournir un dossier de travail unique et partage.
Les fonctions ne dependent pas d'une installation locale d'Apache Airflow.
La regression logistique garde C=1. Un echec avant publication conserve le modele existant.
Les rapports d'execution sont ecrits dans le dossier de travail, pas dans `metrique/`.

Le service Docker Airflow est prepare dans `docker-compose.yml` et `Dockerfile.airflow`.
Le fichier DAG lui-meme reste a ajouter dans `dags/` pour definir le planning et les reprises.
Ne pas presenter cette configuration comme une orchestration deja complete.
Les anciens exports dans `reports/` sont facultatifs et exclus de Git.
