from typing import Final
import numpy as np
import pandas as pd
from numba import njit


@njit
def impurity_numba(counts: np.ndarray) -> float:
    """Numba-friendly Gini impurity from a 1D counts array."""
    total = 0
    for i in range(counts.size):
        total += counts[i]
    if total == 0:
        return 0.0
    s = 0.0
    for i in range(counts.size):
        p = counts[i] / total
        s += p * p
    return 1.0 - s

@njit
def gini_gain_numba(attr: np.ndarray, y: np.ndarray) -> float:
    """
    Gini gain for splitting labels y by categorical attribute attr.
    Pure Numba; counts via loops; calls gini_impurity_from_counts for parent/children.

    Requirements:
      - attr: int array of shape (n,), values in 0..k-1
      - y:    int array of shape (n,), values in 0..c-1
    """
    n = attr.size
    if n == 0:
        return 0.0

    # Infer k and c from maxima (dense encoding assumed)
    max_v = 0
    max_c = 0
    for i in range(n):
        if attr[i] > max_v:
            max_v = attr[i]
        if y[i] > max_c:
            max_c = y[i]
    k = max_v + 1
    c = max_c + 1

    # Allocate counters
    counts_parent = np.zeros(c, dtype=np.int64)
    counts_v = np.zeros(k, dtype=np.int64)
    counts_vc = np.zeros((k, c), dtype=np.int64)

    # Single pass to fill counts
    for i in range(n):
        v = attr[i]
        cl = y[i]
        counts_parent[cl] += 1
        counts_v[v]+= 1
        counts_vc[v, cl]+= 1

    # Parent impurity
    parent_gini = impurity_numba(counts_parent)

    # Weighted child impurities
    split_gini = 0.0
    inv_n = 1.0 / n
    for v in range(k):
        nv = counts_v[v]
        if nv == 0:
            continue
        split_gini += (nv * inv_n) * impurity_numba(counts_vc[v])

    return parent_gini - split_gini


def impurity_numpy(counts: np.ndarray) -> float:
    """Gini impurity given class counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.dot(p, p)


def gini_gain_numpy(attr: Iterable, y: Iterable) -> float:
    """
    Gini gain for splitting labels y by categorical attribute attr.
    Pure NumPy; works for any dtype (auto-encodes with np.unique).
    """
    attr = np.asarray(attr)
    y = np.asarray(y)
    if attr.ndim != 1 or y.ndim != 1 or attr.size != y.size:
        raise ValueError("attr and y must be 1D with the same length")

    n = attr.size
    # Parent impurity
    _, counts_parent = np.unique(y, return_counts=True)
    parent_gini = impurity_numpy(counts_parent)

    # Encode to 0..k-1 and 0..c-1
    v_vals, inv_v = np.unique(attr, return_inverse=True)
    c_vals, inv_c = np.unique(y,    return_inverse=True)
    k, c = v_vals.size, c_vals.size

    # Counts per value and per (value, class)
    counts_v  = np.bincount(inv_v, minlength=k)
    pair_idx  = inv_v * c + inv_c
    counts_vc = np.bincount(pair_idx, minlength=k * c).reshape(k, c)

    # Weighted child Gini using the helper
    split_gini = 0.0
    inv_n = 1.0 / n
    for v in range(k):
        n_v = counts_v[v]
        if n_v == 0:
            continue
        split_gini += (n_v * inv_n) * impurity_numpy(counts_vc[v])

    return parent_gini - split_gini






if __name__ == "__main__":    # Example usage
    counts = np.array([5, 3, 2])
    impurity = impurity_numba(counts)
    # print(f"Gini Impurity: {impurity}")
    data = pd.read_csv('../../data/lab_2/playgolf.csv')
    print(data)
    y = data.iloc[:, 4].to_numpy()
    print(y)
    X = data.iloc[:, 0:4].to_numpy()
    print(X.shape)
    print(X[0])
    outlook = X[:, 0]