from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.outlier import IQROutlierTransformer


# Age et annee universitaire sont controles dans le notebook, sans bornage automatique.
OUTLIER_COLUMNS = [
    "Sleep_Hours", "Study_Hours_Per_Week", "GPA", "Social_Media_Hours",
    "Screen_Time_Hours", "Financial_Stress", "Family_Support", "Attendance_Rate",
]


def build_preprocessor(X_train, use_outliers=True):
    """Construire les transformations sans encore apprendre leurs parametres."""
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    category_columns = X_train.select_dtypes(exclude="number").columns.tolist()

    # Branche des nombres : remplir les trous puis centrer et reduire.
    numeric_pipeline = Pipeline([
        ("imputation", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("standardisation", StandardScaler()),
    ])

    # Branche des textes : remplir les trous puis creer des indicateurs 0/1.
    category_pipeline = Pipeline([
        ("imputation", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preparation = ColumnTransformer([
        ("numerique", numeric_pipeline, numeric_columns),
        ("categorie", category_pipeline, category_columns),
    ])

    outliers = "passthrough"
    if use_outliers:
        columns = [column for column in OUTLIER_COLUMNS if column in numeric_columns]
        outliers = IQROutlierTransformer(columns=columns)

    # Le bornage passe avant l'imputation et l'encodage.
    return Pipeline([("outliers", outliers), ("preparation", preparation)])
