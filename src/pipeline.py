from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.outlier import IQROutlierTransformer
from src.features import NumericFeatures


OUTLIER_COLUMNS = [
    "Sleep_Hours", "Study_Hours_Per_Week", "GPA", "Social_Media_Hours",
    "Screen_Time_Hours", "Financial_Stress", "Family_Support", "Attendance_Rate",
]


def build_preprocessor(X_train, use_outliers=True, feature_set="none"):

    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    category_columns = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputation", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("features", NumericFeatures(numeric_columns, feature_set)),
        ("standardisation", StandardScaler()),
    ])

    category_pipeline = Pipeline([
        ("imputation", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preparation = ColumnTransformer([
        ("numerique", numeric_pipeline, numeric_columns),
        ("categorie", category_pipeline, category_columns),
    ])

    outlier_step = "passthrough"
    if use_outliers:
        columns = [column for column in OUTLIER_COLUMNS if column in numeric_columns]
        outlier_step = IQROutlierTransformer(columns=columns)

    return Pipeline([
        ("outliers", outlier_step),
        ("preparation", preparation),
    ])
