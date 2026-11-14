from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.pipeline import build_preprocessor


def train_model(model, X_train, y_train, use_outliers=True):
    """Entrainer un pipeline neuf sur les colonnes brutes du train."""
    pipeline = Pipeline([
        ("preprocessing", build_preprocessor(X_train, use_outliers)),
        ("model", clone(model)),
    ])
    # fit apprend les medianes, les echelles et le classifieur sur le train uniquement.
    pipeline.fit(X_train, y_train)
    return pipeline
