
import math
from functools import cache

import numpy as np
from keras.src.trainers.trainer import Trainer
from numba import njit
import time
import matplotlib.pyplot as plt
from time import perf_counter_ns
""" Lab 1 Task 1: Distance functions solutions:

The exercise involved implementing and benchmarking the Euclidean distance and DTW
distance functions using both pure Python and Numba-accelerated versions.

Measuring distance/similarity is fundamental in many ML algorithms. This task demonstrates why native Python with loops can be slow, and how Numba helps. All these functions take two 1D numpy arrays and return a float. You can assume they are the same length. 
1.	Euclidean distance (O(n)): Implement a function with a single loop (no input checks) that computes the sum of squared differences, then returns the square root.

see euclidean_distance_no_numba(x: np.ndarray, y: np.ndarray) -> float

2.	Numba version: Duplicate the function (with a different name) and decorate with @njit. Remember to set cache=True (see lecture 1.2 slide 40). Write a small test to verify the output is the same. 

see euclidean_distance_numba(x: np.ndarray, y: np.ndarray) -> float

3.	Timing experiment: Use the provided pattern to time both versions over increasing n. Plot time vs n on log–log axes with matplotlib. You will have to make n big as these functions are very fast. There is an example in the repository if you need help (lab_1.py), but its good to get to grips with matplotlib.  

timing_experiment(max_n: int, increment: int, function1, function2) -> np.ndarray:

4.	Dynamic Time Warping (DTW, O(n²)): DTW is a distance measure specifically for time series (see background below). Implement the standard version shown in Algorithm 1. Re-run the timing experiment with and without Numba. Estimate the speed-up for Euclidean and DTW from using Numba.

dtw_distance_no_numba(x: np.ndarray, y: np.ndarray) -> float
dtw_distance_numba(x: np.ndarray, y: np.ndarray) -> float
timing_experiment(max_n: int, increment: int, function1, function2) -> np.ndarray:

"""
def euclidean_distance_no_numba(x: np.ndarray, y: np.ndarray) -> float:
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


@njit(cache=True)
def euclidean_distance_numba(x: np.ndarray, y: np.ndarray) -> float:
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


def dtw_distance_no_numba(x: np.ndarray, y: np.ndarray) -> float:
    """
    Vanilla DTW distance (equal-length series).

    Uses squared-Euclidean local cost ``(x[i]-y[j])**2`` and returns
    ``sqrt`` of the accumulated cost.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional time series of length ``m`` (dtype float64 recommended).
    y : np.ndarray
        One-dimensional time series of length ``m`` (must match ``x``).

    Returns
    -------
    float
        Dynamic Time Warping distance between ``x`` and ``y`` (L2-style).

    Notes
    -----
    Assumes ``len(x) == len(y)`` and no window or path constraints.
    """
    m = x.shape[0]
    C = np.full((m + 1, m + 1), np.inf, dtype=np.float64)
    C[0, 0] = 0.0

    for i in range(1, m + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            d = (xi - y[j - 1]) ** 2
            C[i, j] = d + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])

    return float(np.sqrt(C[m, m]))



@njit(cache=True)
def dtw_distance_numba(x: np.ndarray, y: np.ndarray) -> float:
    """
    Vanilla DTW distance (equal-length series).

    Uses squared-Euclidean local cost ``(x[i]-y[j])**2`` and returns
    ``sqrt`` of the accumulated cost.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional time series of length ``m`` (dtype float64 recommended).
    y : np.ndarray
        One-dimensional time series of length ``m`` (must match ``x``).

    Returns
    -------
    float
        Dynamic Time Warping distance between ``x`` and ``y`` (L2-style).

    Notes
    -----
    Assumes ``len(x) == len(y)`` and no window or path constraints.
    """
    m = x.shape[0]
    C = np.full((m + 1, m + 1), np.inf, dtype=np.float64)
    C[0, 0] = 0.0

    for i in range(1, m + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            d = (xi - y[j - 1]) ** 2
            C[i, j] = d + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])

    return float(np.sqrt(C[m, m]))


@njit(cache=True)
def dtw_distance_numba_optimised(x: np.ndarray, y: np.ndarray) -> float:
    """
    Numba-accelerated vanilla DTW (equal-length series).

    Uses squared-Euclidean local cost ``(x[i]-y[j])**2`` and returns
    ``sqrt`` of the accumulated cost.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional time series of length ``m`` (dtype float64 recommended).
    y : np.ndarray
        One-dimensional time series of length ``m`` (must match ``x``).

    Returns
    -------
    float
        Squared DTW distance between ``x`` and ``y`` (L2-style).

    Notes
    -----
    Assumes ``len(x) == len(y)`` and no window or path constraints.
    """
    m = x.shape[0]
    C = np.empty((m + 1, m + 1), dtype=np.float64)

    inf = np.inf
    for i in range(m + 1):
        for j in range(m + 1):
            C[i, j] = inf
    C[0, 0] = 0.0

    for i in range(1, m + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            d = (xi - y[j - 1]) ** 2
            up = C[i - 1, j]
            left = C[i, j - 1]
            diag = C[i - 1, j - 1]
            mmin = up if up < left else left
            if diag < mmin:
                mmin = diag
            C[i, j] = d + mmin

    return math.sqrt(C[m, m])


def timing_experiment(max_n: int, increment: int, function1, function2) -> np.ndarray:
    """
    Benchmark two functions over increasing input sizes and return wall-clock timings.

    For n = increment, 2*increment, ..., max_n, this routine generates two random
    1-D NumPy arrays of length n (float64), calls each function once, and records
    the elapsed wall-clock time for each call. It also checks that both functions
    produce numerically equivalent outputs on the same inputs using
    ``np.allclose``.

    Parameters
    ----------
    max_n : int
        The maximum input size to test (inclusive). Must be >= increment.
    increment : int
        Step size between successive input lengths. Must be > 0.
    function1, function2 : callable
        Callables with signature ``func(arr1: np.ndarray, arr2: np.ndarray) -> Any``.
        Each will be invoked once per input size with two arrays of shape ``(n,)``
        and dtype ``float64``.

    Returns
    -------
    sizes : np.ndarray
        1-D array of tested sizes ``[increment, 2*increment, ..., max_n]``.
    t1_list : np.ndarray
        1-D array of wall-clock times (seconds) for ``function1`` at each size.
    t2_list : np.ndarray
        1-D array of wall-clock times (seconds) for ``function2`` at each size.

    Raises
    ------
    AssertionError
        If the outputs of ``function1`` and ``function2`` are not close according
        to ``np.allclose`` for any tested input size.

    Examples
    --------
    >>> def f1(a, b): return np.dot(a, b)
    >>> def f2(a, b): return (a * b).sum()
    >>> sizes, t1, t2, d = timing_experiment(100_000, 10_000, f1, f2)
    """
    sizes = []
    t1_list = []
    t2_list = []
    distances =[]
    for n in range(increment, max_n + 1, increment):
        arr1 = np.random.rand(n).astype(np.float64)
        arr2 = np.random.rand(n).astype(np.float64)

        # Time function1
        t0 = time.perf_counter_ns()
        d1 = function1(arr1, arr2)
        t1_list.append(time.perf_counter_ns() - t0)

        # Time function2
        t0 = time.perf_counter_ns()
        d2 = function2(arr1, arr2)
        t2_list.append(time.perf_counter_ns() - t0)
        distances.append((d1,d2))
        sizes.append(n)
    return np.array(sizes), np.array(t1_list), np.array(t2_list), distances


def plot_timing(sizes, t1, t2, labels=("function1", "function2"), y_floor=1e-9):
    """
    Plot timing results with time on a log scale.

    Parameters
    ----------
    sizes : array-like
        Problem sizes (x-axis).
    t1, t2 : array-like
        Timings (seconds) for the two functions.
    labels : tuple[str, str]
        Legend labels for the two curves.
    y_floor : float
        Minimum positive value used to avoid log(0) if any timings are zero.
    """
    sizes = np.asarray(sizes)
    t1 = np.maximum(np.asarray(t1, dtype=float), y_floor)
    t2 = np.maximum(np.asarray(t2, dtype=float), y_floor)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(sizes, t1, marker="o", label=labels[0])
    ax.plot(sizes, t2, marker="s", label=labels[1])
    ax.set_xlabel("n")
    ax.set_ylabel("Time (nanoseconds, log scale)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return ax


if __name__ == "__main__":
    # --- Configuration ---
    max_n = 10_000_000
    increment = 1_000_000

    # Warm-up call to trigger Numba compilation (excluded from timings)
    _ = euclidean_distance_numba(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    _ = dtw_distance_numba(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    _ = dtw_distance_numba_optimised(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    # Run timing experiment: ED
    sizes, t1, t2, distances = timing_experiment(
        max_n=max_n,
        increment=increment,
        function1=euclidean_distance_no_numba,
        function2=euclidean_distance_numba,
    )

    # Plot (log-scale y)
    plot_timing(sizes, t1, t2, labels=("Python", "Numba @njit"))
    plt.title("DTW distance: Python vs Numba")
    plt.show()
    print(" ED numba speed up = ", np.average(t1 / t2))

    max_n = 1000
    increment = 100

    # Run timing experiment: DTW
    sizes, t1, t2, distances = timing_experiment(
        max_n=max_n,
        increment=increment,
        function1=dtw_distance_no_numba,
        function2=dtw_distance_numba,
    )

    # Plot (log-scale y)
    plot_timing(sizes, t1, t2, labels=("Python", "Numba @njit"))
    plt.title("DTW distance: Python vs Numba")
    plt.show()
    print(" DTW Numba speed up = ", np.average(t1 / t2))


    # Run timing experiment: DTW
    sizes, t1, t2, distances = timing_experiment(
        max_n=max_n,
        increment=increment,
        function1=dtw_distance_numba,
        function2=dtw_distance_numba_optimised,
    )

    # Plot (log-scale y)
    plot_timing(sizes, t1, t2, labels=("Numba basic", "Numba optimised"))
    plt.title("DTW distance: Numba vs Numba optimised")
    plt.show()
    print(" DTW optimised Numba speed up = ", np.average(t1 / t2))