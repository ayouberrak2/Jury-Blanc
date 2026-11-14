import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.pipeline import build_preprocessor
from src.evaluate import evaluate_predictions


def optimize_model(model, parameters, X_train, y_train):
    """Choisir les parametres sur 3 plis et reentrainer sur les donnees recues."""
    pipeline = Pipeline([
        ("preprocessing", build_preprocessor(X_train)),
        ("model", clone(model)),
    ])
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline, parameters, cv=inner_cv, scoring="f1_macro",
        n_jobs=1, return_train_score=True, error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def cross_validate_model(model, parameters, X_train, y_train, class_names):
    """Evaluer l'optimisation sur 5 plis externes, sans toucher au test."""
    y_train = np.asarray(y_train)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.zeros(len(X_train), dtype=int)
    probabilities = np.zeros((len(X_train), len(class_names)))
    fold_numbers = np.zeros(len(X_train), dtype=int)
    rows = []

    for fold, (train_index, val_index) in enumerate(outer_cv.split(X_train, y_train), start=1):
        X_fit = X_train.iloc[train_index]
        X_val = X_train.iloc[val_index]
        y_fit = y_train[train_index]
        y_val = y_train[val_index]

        # Le reglage interne ne voit pas les lignes de validation de ce pli.
        search = optimize_model(model, parameters, X_fit, y_fit)
        best_model = search.best_estimator_
        fold_predictions = best_model.predict(X_val)
        fold_probabilities = best_model.predict_proba(X_val)
        predictions[val_index] = fold_predictions
        probabilities[val_index] = fold_probabilities
        fold_numbers[val_index] = fold

        result = evaluate_predictions(y_val, fold_predictions, class_names, fold_probabilities)
        row = {"Fold": fold, "Accuracy train": best_model.score(X_fit, y_fit)}
        row.update(result["metrics"])
        rows.append(row)
        print("Pli", fold, "termine", flush=True)

    # Chaque ligne du train possede maintenant une prediction hors pli.
    return {
        "evaluation": evaluate_predictions(y_train, predictions, class_names, probabilities),
        "folds": pd.DataFrame(rows),
        "probabilities": probabilities,
        "fold_numbers": fold_numbers,
    }
