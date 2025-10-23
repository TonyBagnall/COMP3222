import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple
from typing import Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
""" Lab 1 Task 2: Data Preprocessing with scikit-learn

Show how to use one hot encoding, imputation of missing values, normalisation,
and pipelines in scikit-learn.
A.	Categorical variables
one_hot_categorical(): One-hot encode nominal features and combine with numeric features.
B.	Missing values
impute_missing(): Impute missing values using SimpleImputer.
C.	Normalisation
normalise(): Standardise features to zero mean and unit variance.
D.	Pipeline
make_pipeline(): Create a scikit-learn Pipeline chaining imputation, scaling, and a classifier.
E.	Train/Test split + quick fit/evaluate helper
fit_evaluate(): Split, fit a default pipeline, and report accuracy.



"""
# --- 1) Load the data (pandas → NumPy) ---
path = "../../data/one_hot.csv"   # adjust to where you saved it
df_onehot = pd.read_csv(
    path,
    header=None,
    names=["x1", "x2", "x3", "rank", "colour", "y"]
)
# continuous features (n, 3)
X_cont = df_onehot[["x1", "x2", "x3"]].to_numpy(dtype=float)
# ordinal feature as numeric (n,)
rank = df_onehot["rank"].to_numpy(dtype=float)
# nominal feature as strings (n,)
colour = df_onehot["colour"].to_numpy(dtype=str)
# labels (n,)
y = df_onehot["y"].to_numpy(dtype=int)

# Suppose your CSV has features in all but the last column, labels in the last:
df_missing = pd.read_csv("../../data/missing.csv", header=None)
X_raw = df_missing.iloc[:, :-1].to_numpy()
y = df_missing.iloc[:, -1].to_numpy()


def one_hot_categorical():
    # --- 2) Combine numeric features (make X_num purely numeric) ---
    # Append rank as a float column to the continuous block → (n, 4)
    X_num = np.column_stack([X_cont, rank])

    # --- 3) One-hot encode the nominal colour ---
    # OneHotEncoder expects 2D input → reshape to (n, 1)
    colour_2d = colour.reshape(-1, 1)

    # NOTE: use sparse=False for broad scikit-learn compatibility
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=float)
    X_cat = ohe.fit_transform(colour_2d)  # shape (n, n_colours)
    print("Shape after one-hot:", X_cat.shape)
    print("Categories:", ohe.categories_)

    # --- 4) Final feature matrix: numeric + one-hot ---
    X = np.column_stack([X_num, X_cat])   # shape (n, 4 + n_colours)

    print("Shapes → X:", X.shape, "  y:", y.shape)

    # --- 5) Sanity check: fit a simple classifier ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"DecisionTree accuracy: {acc:.3f}")



# --- B) Missing values: impute with SimpleImputer ---
def impute_missing(X: np.ndarray, strategy: str = "median") -> Tuple[np.ndarray, SimpleImputer]:
    """
    Impute missing values in X using the given strategy.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix with NaNs or empty cells.
    strategy : {"mean", "median", "most_frequent", "constant"}
        Imputation strategy.

    Returns
    -------
    X_imp : np.ndarray
        Imputed feature matrix.
    imputer : SimpleImputer
        Fitted imputer (use `imputer.transform` for new data).
    """
    imputer = SimpleImputer(strategy=strategy)
    X_imp = imputer.fit_transform(X)
    return X_imp, imputer


# --- C) Normalisation: StandardScaler ---
def normalise(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardise features to zero mean and unit variance.

    Returns
    -------
    X_std : np.ndarray
        Scaled features.
    scaler : StandardScaler
        Fitted scaler.
    """
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std, scaler


# --- D) Pipeline: Imputer → Scaler → Classifier ---
def make_pipeline(
    impute_strategy: str = "median",
    clf: DecisionTreeClassifier | None = None,
) -> Pipeline:
    """
    Create a scikit-learn Pipeline chaining imputation, scaling, and a classifier.
    """
    if clf is None:
        clf = DecisionTreeClassifier(random_state=0)
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy=impute_strategy)),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])
    return pipe


# --- E) Train/Test split + quick fit/evaluate helper ---
def fit_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
) -> dict:
    """
    Split, fit a default pipeline, and report accuracy.

    Returns
    -------
    result : dict with keys ["pipeline", "X_train", "X_test", "y_train", "y_test", "y_pred", "accuracy"]
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=y
    )
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {
        "pipeline": pipe,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "accuracy": acc,
    }


if __name__ == "__main__":
    # Adjust path as needed
    path = "../../data/missing.csv"

    # Load: features = all but last column, labels = last column
    df = pd.read_csv(path, header=None)
    X_raw = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    print(f"Loaded X_raw {X_raw.shape}, y {y.shape}")

    # B) Impute missing values
    X_imp, imputer = impute_missing(X_raw, strategy="median")
    print("After imputation:", X_imp.shape, "| any NaNs left?", np.isnan(X_imp).any())

    # C) Normalise (on the imputed matrix)
    X_std, scaler = normalise(X_imp)
    print("After normalisation:", X_std.shape,
          "| per-feature mean≈0?", np.allclose(X_std.mean(axis=0), 0, atol=1e-8),
          "| std≈1?", np.allclose(X_std.std(axis=0, ddof=0), 1, atol=1e-6))

    # D+E) Pipeline with train/test split (imputer + scaler inside the pipeline)
    result = fit_evaluate(X_imp, y)  # we pass X_imp; pipeline re-imputes/scales safely
    print(f"Pipeline classifier: {result['pipeline'][-1].__class__.__name__}")
    print(f"Held-out accuracy: {result['accuracy']:.3f}")
