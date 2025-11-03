from typing import Final
import numpy as np
import pandas as pd
from numba import njit
""" Lab 2 Task 1: Gini Impurity and Gini Gain

implement it as a simple python function to find the Gini index and gini gain,
then test it using the toy example we used in the lecture.


gini_impurity_numba and gini_impurity_numpy(counts: np.ndarray) -> float
these are simple functions that compute the Gini impurity given class counts. They 
are applicable to a single node (parent or child). It is unlikely there will be much 
speed difference here, because numpy is efficient at these operations.

gini_gain_numba and gini_gain_numpy(attr: np.ndarray, y: np.ndarray) -> float:
These take a single attribute, split the class variable y by the attribute values, 
then call gini_impurity on the parent and children to compute the Gini gain. 

The numpy version uses np functions np.unique and np.bincount.
https://numpy.org/devdocs/reference/generated/numpy.unique.html
https://numpy.org/doc/2.1/reference/generated/numpy.bincount.html
    v_vals, inv_v = np.unique(attr, return_inverse=True)
recovers the unique attribute values v_vals (length v) and their encoding for each 
case: inv_v is an array of length n. 
    c_vals, inv_c = np.unique(y,    return_inverse=True)
does the same for the class labels y, giving c_vals (length c) and inv_c (length n).

The more confusing line is this to form the contingency table. 
    # Counts per value
    counts_v  = np.bincount(inv_v, minlength=k)
gives the number of cases per attribute value (k is the number of unique attribute 
values).  we want the counts per (value, class) pair in a 2D array of shape (k, c). 
With a loop, assuming we map onto 0...k-1 and 0...v-1, this would be:
# counts_vc[v, cls] = number of samples with attr value v and class cls
counts_vc = np.zeros((k, c), dtype=int)
for i in range(n):
    counts_vc[inv_v[i], inv_c[i]] += 1   
But loops are bad in python. Instead,The trick is to use numpy bincount again on a 
flattened 1D index. We create a new index that shows exactly which cell each element 
belongs to. This is called a row major mapping of 2D to 1D. Class index (inv_c) gives 
the column, and the row is inv_v, which we need to scale by c (the number of columns) 
    pair_idx  = inv_v * c + inv_c
We then count how many are in each of these cells using bincount, and reshape back to 
shape(k,c) 
    counts_vc = np.bincount(pair_idx, minlength=k * c).reshape(k, c)
this may seem over complex, but it is very efficient in numpy and is a common pattern
in Python.

The numba version just uses loops throughout. 

"""


def gini_impurity_numpy(counts: np.ndarray) -> float:
    """Gini impurity given class counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.dot(p, p)


@njit(cache=True)
def gini_impurity_numba(counts: np.ndarray) -> float:
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


def gini_gain_numpy(attr: np.ndarray, y: np.ndarray) -> float:
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
    parent_gini = gini_impurity_numpy(counts_parent)

    # Encode attributes to 0..k-1 to indicate values and y into 0..c-1
    # v_vals is an array of length v (number of attribute values
    # inv_c  is an array of length n with class indices 0..c-1
    # shape (n,)
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
        split_gini += (n_v * inv_n) * gini_impurity_numpy(counts_vc[v])

    return parent_gini - split_gini


@njit(cache=True)
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
    parent_gini = gini_impurity_numba(counts_parent)

    # Weighted child impurities
    split_gini = 0.0
    inv_n = 1.0 / n
    for v in range(k):
        nv = counts_v[v]
        if nv == 0:
            continue
        split_gini += (nv * inv_n) * gini_impurity_numba(counts_vc[v])

    return parent_gini - split_gini





if __name__ == "__main__":    # Example usage
    counts = np.array([5, 3, 2])
    impurity = gini_impurity_numba(counts)
    # print(f"Gini Impurity: {impurity}")
    data = pd.read_csv('../../data/lab2/playgolf.csv')
    print(data)
    y = data.iloc[:, 4].to_numpy()
    print(y)
    X = data.iloc[:, 0:4].to_numpy()
    print(X.shape)
    print(X[0])
    outlook = X[:, 0]