import numpy as np
from dataclasses import dataclass
from typing import Any, Iterable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
""" Lab 1 Part 3: Implementing a Custom Classifier in scikit-learn

MedianThresholdRule
implements a simple binary classification rule based on a single feature
Constructor
fit
predict

"""

class MedianThresholdRule(BaseEstimator, ClassifierMixin):
    """
    A toy, scikit-learn-compatible binary classifier that uses a single feature
    and a fixed threshold.

    Rule learned in `fit`:
        1) Take column `feature_index` from X.
        2) Compute the mean of this feature for each class (y == class_0 / class_1).
        3) The class with the *smaller* mean becomes the "less-than" class.
        4) Set the decision threshold to the *median* of the feature over all training rows.
        5) Decision at inference: if x[:, feature_index] < threshold → less_class else other_class.

    Parameters
    ----------
    feature_index : int, default=0
        Column index of the feature to use.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Class labels seen during `fit`.
    n_features_in_ : int
        Number of features seen during `fit`.
    threshold_ : float
        Median threshold of the chosen feature on the training data.
    less_class_ : object
        Label predicted when the feature value is < threshold_.
    greater_class_ : object
        Label predicted when the feature value is >= threshold_.
    """

    def __init__(self, feature_index: int = 0):
        self.feature_index = feature_index

    # ---- scikit-learn API ----
    def fit(self, X: np.ndarray, y: Iterable[Any]):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.number)
        self.n_features_in_ = X.shape[1]

        if not (0 <= self.feature_index < self.n_features_in_):
            raise ValueError(
                f"feature_index={self.feature_index} is out of bounds for X with "
                f"{self.n_features_in_} features."
            )

        # Binary only
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("MedianThresholdRule supports only two classes.")

        # Compute class-wise means for the chosen feature
        col = X[:, self.feature_index]
        mean_per_class = {}
        for c in self.classes_:
            mean_per_class[c] = float(col[y == c].mean())

        # Determine which class is the "less-than" side
        # (the one with the smaller mean)
        c0, c1 = self.classes_
        self.less_class_ = c0 if mean_per_class[c0] < mean_per_class[c1] else c1
        self.greater_class_ = c1 if self.less_class_ == c0 else c0

        # Threshold is the median of the feature across all training data
        self.threshold_ = float(np.median(col))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, accept_sparse=False, dtype=np.number)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this estimator was fitted with "
                f"{self.n_features_in_}."
            )
        col = X[:, self.feature_index]
        preds = np.where(col < self.threshold_, self.less_class_, self.greater_class_)
        return preds.astype(self.classes_.dtype, copy=False)


# ---- Minimal demo ---------------------------------------------------------
if __name__ == "__main__":
    # Toy data: two classes that differ on column 0
    rng = np.random.default_rng(0)
    X0 = np.c_[rng.normal(-1.0, 0.5, 80), rng.normal(0.0, 1.0, 80), rng.normal(0.0, 1.0, 80)]
    X1 = np.c_[rng.normal(+1.0, 0.5, 80), rng.normal(0.0, 1.0, 80), rng.normal(0.0, 1.0, 80)]
    X = np.vstack([X0, X1])
    y = np.array([0]*len(X0) + [1]*len(X1))

    clf = MedianThresholdRule(feature_index=0).fit(X, y)
    yhat = clf.predict(X)
    acc = (yhat == y).mean()
    print(f"threshold={clf.threshold_:.3f}, less_class={clf.less_class_}, "
          f"greater_class={clf.greater_class_}, accuracy={acc:.3f}")
