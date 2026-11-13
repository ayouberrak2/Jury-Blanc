from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class IQROutlierTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, columns, factor=1.5):
        self.columns = columns
        self.factor = factor

    def fit(self, X, y=None):

        if self.factor <= 0:
            raise ValueError("factor doit etre positif")
        self.limits_ = {}
        for column in self.columns:
            q1 = X[column].quantile(0.25)
            q3 = X[column].quantile(0.75)
            iqr = q3 - q1

            if iqr > 0:
                lower = q1 - self.factor * iqr
                upper = q3 + self.factor * iqr
                self.limits_[column] = (lower, upper)
        return self

    def transform(self, X):
        check_is_fitted(self, "limits_")
        result = X.copy()
        for column, (lower, upper) in self.limits_.items():

            result[column] = result[column].clip(lower=lower, upper=upper)
        return result
