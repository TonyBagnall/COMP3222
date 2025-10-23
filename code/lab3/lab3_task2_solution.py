from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted
""" Lab 3 task 2: I am not giving a full solution for this part, since it is too like 
the coursework. I have limited it to the constructor and docstring only."""

class BasicEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    A very simple majority-vote ensemble.

    Parameters
    ----------
    base_estimator : estimator
        Any sklearn-compatible classifier (must implement fit/predict; predict_proba optional).
    n_estimators : int, default=10
        Number of cloned members.
    sample_fraction : float, default=0.6
        Fraction of training samples used per member (rounded to at least 1).
    replace : bool, default=False
        If True, sample with replacement (bootstrap). Default is without replacement.
    random_state : int or None, default=None
        Seed for reproducibility.

    Notes
    -----
    - Cloning uses sklearn.base.clone.
    - Sampling uses numpy choice for speed and clarity.
    """

    def __init__(self, base_estimator, n_estimators=100, sample_fraction=0.6,
                 replace=False, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.replace = replace
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        pass

    def predict(self, X):
        pass