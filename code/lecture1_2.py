import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt

def numpy_examples():
    # Create a 1D array
    arr1d = np.array([1, 2, 3, 4, 5])
    print("1D Array:", arr1d)
    # Create a 2D array
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    print("2D Array:\n", arr2d)
    # Array operations
    arr_sum = arr1d + 10
    print("Array after adding 10:", arr_sum)
    arr_mult = arr2d * 2
    print("Array after multiplying by 2:\n", arr_mult)
    # Slicing arrays
    slice_1d = arr1d[1:4]
    print("Sliced 1D Array (index 1 to 3):", slice_1d)
    slice_2d = arr2d[:, 1]
    print("Sliced 2D Array (all rows, column 1):", slice_2d)
    # Reshaping arrays
    reshaped = arr1d.reshape((5, 1))
    print("Reshaped Array (5x1):\n", reshaped)
    # Statistical operations
    mean_val = np.mean(arr1d)
    print("Mean of 1D Array:", mean_val)
    std_val = np.std(arr2d)
    print("Standard Deviation of 2D Array:", std_val)
    # Random numbers
    random_arr = np.random.rand(3, 3)
    print("Random 3x3 Array shape:\n", random_arr)
    # broadcasting
    arr_broadcast=np.ones((3, 4)) + np.array([10, 20, 30, 40])   # (3,4) + (4,) → (3,4)
    print("Broadcasted Array:", arr_broadcast)


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
def scikit_example():
    # Load dataset
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    print("Data type:", type(X), type(y), " Shape:", X.shape, y.shape)
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create a classifier
    model = DecisionTreeClassifier()
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)


def tensorflow_example():
    # 1) Load data
    iris = datasets.load_iris()
    X = iris.data.astype(np.float32)     # shape (150, 4)
    y = iris.target.astype(np.int32)     # labels 0..2
    class_names = iris.target_names
    # 2) Train/test split + scaling (fit scaler on train only)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define a simple model

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # 4) Train
    history = model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    probs = model.predict(X_test, verbose=0) # Output probabilities not predctions
    print(" Type of probs:", type(probs), " Shape:", probs.shape)
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

def torch_example(random_state: int = 0):
    # 1) Data
    iris = datasets.load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)  # CrossEntropyLoss expects int64 class indices
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for epoch in range(30):
        for xb, yb in train_loader:
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)

        acc = (preds == y_test_t).float().mean().item()
        print(f"Test accuracy: {acc:.3f}")


# jit_speed_demo.py
import time
from numba import njit
from statistics import median
# --- sum of squares ---
def sumsq_py(arr: np.ndarray) -> int:
    s = 0
    n = len(arr)
    for i in range(n):
        s += arr[i] * arr[i]
    return s

@njit(cache=True)
def sumsq_nb(arr: np.ndarray) -> float:
    s = 0.0
    n = len(arr)
    for i in range(n):
        s += arr[i] * arr[i]
    return s


def time_once(fn, arr, repeat=3):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(arr)
        times.append(time.perf_counter() - t0)
    return median(times)

def numba_example():
    # Warm-up compile (first call includes JIT compile time)
    arr = np.random.rand(100).astype(np.float64)
    sumsq_nb(arr)
    sizes = []
    t_py_list = []
    t_nb_list = []
    for n in range(100_000, 1000_001, 100_000):
        arr = np.random.rand(n).astype(np.float64)
        t_py = time_once(sumsq_py, arr)
        t_nb = time_once(sumsq_nb, arr)
        speed = t_py / t_nb
        print(f"{n:>10,d}  {t_py:12.6f}  {t_nb:11.6f}  {speed:8.2f}")
        sizes.append(n)
        t_py_list.append(t_py)
        t_nb_list.append(t_nb)

    # --- Plot: time vs n (log–log) ---
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(sizes, t_py_list, marker="o", label="Python")
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

    print("\nNotes:")
    print("• First numba call pays a one-off compile cost (we warmed it up).")
    print("• Expect speed-ups to grow with n (often 10–100× for loopy code).")
    print("• Use `@njit` on numerical, type-stable code; avoid Python objects.")



if __name__ == "__main__":
    # numpy_examples()
    # scikit_examples()
    # tensorflow_example()
    # torch_example()
    numba_example()

