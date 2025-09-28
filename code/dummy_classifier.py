from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import numpy as np

class MyClassifier(BaseEstimator, ClassifierMixin):
    """A simple classifier that separates two classes based on a single attribute.

    Parameters
    ----------
    attribute_index : int, default=0
        Index of the attribute to use for classification.

    Attributes
    ----------
    threshold_ : float
        The threshold value for the decision boundary.
    """

    def __init__(self, attribute_index: int = 0):
        self.attribute_index = attribute_index
        self.threshold_ = 0

    def fit(self,X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        # Classifier training logic here
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=None)
        # Predict class labels for samples in X
        return np.zeros(X.shape[0], dtype=int)
