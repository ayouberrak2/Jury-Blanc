"""Fonctions a importer avec : from src.airflow import ...

Chaque execution du DAG doit avoir son propre work_dir, partage entre les taches.
Les fonctions renvoient ce chemin : Airflow ne transporte pas les tableaux dans XCom.
Aucun import d'Apache Airflow n'est necessaire dans ce fichier.
"""
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.pipeline import build_preprocessor
from src.evaluate import evaluate_model, evaluate_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/raw/student_mental_health.csv"


def prepare_data(work_dir, csv_path=DATA_PATH):
    """Charger, nettoyer les doublons et separer 80 % train / 20 % test."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    data_path = work_dir / "data.joblib"
    # En reprise, conserver le meme snapshot de donnees pour cette execution.
    if data_path.exists():
        previous = joblib.load(data_path)
        if not isinstance(previous["encoder"], OrdinalEncoder):
            raise ValueError("Ancien encodage : utiliser un nouveau dossier pour cette execution")
        return str(work_dir)

    df = pd.read_csv(csv_path)
    required = {"Student_ID", "Mental_Health_Risk"}
    if not required.issubset(df.columns):
        raise ValueError("Student_ID ou Mental_Health_Risk absent du CSV")
    df = df.drop_duplicates().dropna(subset=["Mental_Health_Risk"])
    X = df.drop(columns=["Student_ID", "Mental_Health_Risk"])
    y = df["Mental_Health_Risk"]
    if set(y.unique()) != {"Low", "Medium", "High"}:
        raise ValueError("La cible doit contenir Low, Medium et High")
    X_train, X_test, y_train_text, y_test_text = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    if y_train_text.value_counts().min() < 5 or y_test_text.nunique() != 3:
        raise ValueError("Pas assez de lignes par classe pour evaluer sur 5 plis")
    encoder = OrdinalEncoder(categories=[["Low", "Medium", "High"]]).fit(y_train_text.to_frame())
    joblib.dump({
        "X_train": X_train, "X_test": X_test,
        "y_train": encoder.transform(y_train_text.to_frame()).astype(int).ravel(),
        "y_test": encoder.transform(y_test_text.to_frame()).astype(int).ravel(), "encoder": encoder,
    }, data_path)
    print("Preparation terminee")
    return str(work_dir)


def preprocess_data(work_dir):
    """Apprendre le preprocessing sur le train et transformer train et test."""
    work_dir = Path(work_dir)
    data = joblib.load(work_dir / "data.joblib")
    preprocessor = build_preprocessor(data["X_train"])
    X_train = preprocessor.fit_transform(data["X_train"])
    X_test = preprocessor.transform(data["X_test"])
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Valeurs manquantes ou infinies apres preprocessing")
    joblib.dump({"preprocessor": preprocessor, "X_train": X_train, "X_test": X_test},
                work_dir / "preprocessing.joblib")
    print("Pretraitement termine")
    return str(work_dir)


def train_selected_model(work_dir):
    """Entrainer uniquement la regression logistique deja retenue."""
    work_dir = Path(work_dir)
    data = joblib.load(work_dir / "data.joblib")
    prepared = joblib.load(work_dir / "preprocessing.joblib")
    model = LogisticRegression(C=1, max_iter=2000, random_state=42)
    model.fit(prepared["X_train"], data["y_train"])
    # Assembler les deux etapes deja ajustees pour predire sur les colonnes brutes.
    pipeline = Pipeline([("preprocessing", prepared["preprocessor"]), ("model", model)])
    joblib.dump({
        "model_name": "Logistic Regression", "pipeline": pipeline,
        "target_encoder": data["encoder"], "class_names": data["encoder"].categories_[0],
        "training_rows": len(data["X_train"]), "best_parameters": {"model__C": 1},
        "selection_metric": "Configuration retenue auparavant en validation",
    }, work_dir / "candidate.joblib")
    print("Entrainement termine : candidat pas encore publie")
    return str(work_dir)


def evaluate_and_save(work_dir, project_dir=PROJECT_ROOT):
    """Evaluer puis publier le seul pipeline retenu si les calculs reussissent."""
    work_dir = Path(work_dir)
    data = joblib.load(work_dir / "data.joblib")
    bundle = joblib.load(work_dir / "candidate.joblib")
    pipeline = bundle["pipeline"]
    classes = bundle["class_names"]

    # Parametres figes : CV simple a 5 plis, sans nouvelle optimisation sur le test.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = cross_val_predict(
        clone(pipeline), data["X_train"], data["y_train"],
        cv=cv, method="predict_proba", n_jobs=1,
    )
    results = {
        "Train": evaluate_model(pipeline, data["X_train"], data["y_train"], classes),
        "Validation CV": evaluate_predictions(data["y_train"], probabilities.argmax(axis=1), classes, probabilities),
        "Test": evaluate_model(pipeline, data["X_test"], data["y_test"], classes),
    }
    rows = []
    for split, result in results.items():
        if not np.isfinite(list(result["metrics"].values())).all():
            raise ValueError("Metriques non finies : " + split)
        rows.append({"Modele": bundle["model_name"], "Ensemble": split, **result["metrics"]})
    pd.DataFrame(rows).to_csv(work_dir / "metrics.csv", index=False)
    results["Test"]["confusion_matrix"].to_csv(work_dir / "confusion_test.csv")
    results["Test"]["report"].to_csv(work_dir / "report_test.csv")

    # Ne remplacer le modele existant qu'apres une evaluation reussie.
    model_dir = Path(project_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    temporary = model_dir / ("selected_model_" + uuid.uuid4().hex + ".tmp")
    try:
        joblib.dump(bundle, temporary)
        temporary.replace(model_dir / "selected_model.joblib")
    finally:
        temporary.unlink(missing_ok=True)
    (work_dir / "candidate.joblib").unlink()
    print("Evaluation terminee : seul selected_model.joblib est publie")
    return str(work_dir)
