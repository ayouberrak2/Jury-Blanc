import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def evaluate_predictions(y_validation, predictions, class_names, probabilities=None):
    labels = np.arange(len(class_names))

    report = pd.DataFrame(classification_report(
        y_validation, predictions, labels=labels,
        target_names=list(class_names), output_dict=True, zero_division=0,
    )).T

    macro = report.loc["macro avg"]
    metrics = {
        "Accuracy": accuracy_score(y_validation, predictions),
        "Precision macro": macro["precision"],
        "Recall macro": macro["recall"],
        "F1 macro": macro["f1-score"],
        "ROC AUC macro OvR": np.nan,
    }
    if probabilities is not None and len(np.unique(y_validation)) == len(labels):
        metrics["ROC AUC macro OvR"] = roc_auc_score(
            y_validation, probabilities,
            labels=labels, multi_class="ovr", average="macro",
        )

    matrix = pd.DataFrame(
        confusion_matrix(y_validation, predictions, labels=labels),
        index=class_names, columns=class_names,
    )
    matrix.index.name, matrix.columns.name = "Reel", "Predit"
    return {"metrics": metrics, "report": report,
            "confusion_matrix": matrix, "predictions": predictions}


def evaluate_model(model, X_validation, y_validation, class_names):
    return evaluate_predictions(
        y_validation, model.predict(X_validation), class_names,
        model.predict_proba(X_validation),
    )
