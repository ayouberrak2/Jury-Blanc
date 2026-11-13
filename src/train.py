from sklearn.base import clone
from sklearn.pipeline import Pipeline
from src.pipeline import build_preprocessor


def train_model(model, X_train, y_train, use_outliers=True):

    pipeline = Pipeline([
        ("preprocessing", build_preprocessor(X_train, use_outliers)),
        ("model", clone(model)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
