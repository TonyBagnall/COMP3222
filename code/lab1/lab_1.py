import numpy as np
import math


def euclidean_distance_python(x: np.ndarray, y: np.ndarray) -> float:
    r"""Compute the Euclidean distance between two time series.

    The Euclidean distance between two time series of length m is the square root of
    the squared distance and is defined as:

    .. math::
        ed(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}

    This implementation assumes that both time series are of equal length. It uses
    native python to perform the calculation.

    Parameters
    ----------
    x : 1D np.ndarray
        First time series
    y : 1D np.ndarray
        Second time series of equal length to x

    Returns
    -------
    float
        Euclidean distance between x and y.
    """
    distance = 0
    for i in range(len(x)):
        difference = x[i] - y[i]
        distance += difference * difference
    return math.sqrt(distance)

def part1_example():
    # Basic usage
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10,11])
    print(euclidean_distance(x, y))

    m = len(x)
    C = np.full((m + 1, m + 1), np.inf, dtype=np.float64)

    # Time for increasing sizes of input
    sizes = []
    ed_list = []
    for n in range(100_000, 1000_001, 100_000):
        arr1 = np.random.rand(n).astype(np.float64)
        arr2 = np.random.rand(n).astype(np.float64)
        t0 = time.perf_counter()
        euclidean_distance(arr1,arr2)
        ed_list.append(time.perf_counter() - t0)
    # --- Plot: time vs n (log–log) ---
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(sizes, t_py_list, marker="o", label="ED Python")
    plt.plot(sizes, t_nb_list, marker="s", label="Numba @njit")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n (log scale)")
    plt.ylabel("Time (seconds, log scale)")
    plt.title("Sum of squares: Python vs Numba (time vs n)")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # Save and/or show
    plt.savefig("jit_speed_time_vs_n.png", dpi=150)
    plt.show()

import pandas as pd
def one_hot_example():
    path ="../../data/one_hot.csv"
    df = pd.read_csv(path, header=None, names=["x1","x2","x3",
                                                               "rank","colour", "y"])
    X = df[["x1","x2","x3"]].to_numpy(dtype=float)   # shape (n, 3)
    score = df["rank"].to_numpy(dtype=int)              # shape (n,)
    colour = df["colour"].to_numpy(dtype=str)
    y = df["y"].to_numpy(dtype=int)
    print(X.shape, score.shape, y.shape,colour.shape)
    print(type(X), type(score), type(y),type(colour))


def missing_example():
    path ="../../data/missing.csv"
    X= pd.read_csv(path, header=None).to_numpy(dtype=float)
    print(X)

missing_example()



