import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def evaluate_predictions(y_true, predictions, class_names, probabilities=None):
    """Calculer les scores a partir de predictions deja obtenues."""
    labels = np.arange(len(class_names))
    report = classification_report(
        y_true, predictions, labels=labels,
        target_names=list(class_names), output_dict=True, zero_division=0,
    )
    report = pd.DataFrame(report).T
    macro = report.loc["macro avg"]
    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision macro": macro["precision"],
        "Recall macro": macro["recall"],
        "F1 macro": macro["f1-score"],
        "ROC AUC macro OvR": np.nan,
    }
    # Cette AUC multiclasses exige les probabilites et toutes les classes presentes.
    if probabilities is not None and len(np.unique(y_true)) == len(labels):
        metrics["ROC AUC macro OvR"] = roc_auc_score(
            y_true, probabilities, labels=labels, multi_class="ovr", average="macro",
        )
    matrix = pd.DataFrame(
        confusion_matrix(y_true, predictions, labels=labels),
        index=class_names, columns=class_names,
    )
    matrix.index.name = "Reel"
    matrix.columns.name = "Predit"
    return {"metrics": metrics, "report": report,
            "confusion_matrix": matrix, "predictions": predictions}


def evaluate_model(model, X_evaluation, y_evaluation, class_names):
    """Predire et evaluer un pipeline deja entraine, sans refaire fit."""
    return evaluate_predictions(
        y_evaluation, model.predict(X_evaluation), class_names,
        model.predict_proba(X_evaluation),
    )
