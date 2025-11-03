from aeon.classification.sklearn import RotationForestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from aeon.visualisation import plot_critical_difference
import matplotlib.pyplot as plt

datasets =[
    "bank",
    "blood",
    "breast-cancer-wisc-diag",
    "breast-tissue",
    "cardiotocography-10clases",
    "conn-bench-sonar-mines-rocks",
    "conn-bench-vowel-deterding",
    "ecoli",
    "glass",
    "hill-valley",
    "image-segmentation",
    "ionosphere",
    "iris",
    "libras",
    "oocytes_merluccius_nucleus_4d",
    "oocytes_trisopterus_states_5b",
    "optical",
    "ozone",
    "page-blocks",
    "parkinsons",
    "planning",
    "post-operative",
    "ringnorm",
    "seeds",
    "spambase",
    "statlog-landsat",
    "statlog-vehicle",
    "steel-plates",
    "synthetic-control",
    "twonorm",
    "vertebral-column-3clases",
    "wall-following",
    "waveform-noise",
    "wine-quality-white",
    "yeast",
]

import numpy as np
from scipy.io import arff
from typing import Tuple
def load_arff_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an ARFF file into NumPy arrays (X, y).
    Assumptions:
      - Last column is the target y.
      - All feature columns are real-valued (cast to float64).
      - y may be nominal; bytes are decoded to str.

    Parameters
    ----------
    path : str
        Path to the .arff file.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (float64).
    y : np.ndarray, shape (n_samples,)
        Target vector (strings if nominal; numeric otherwise).
    """
    data, meta = arff.loadarff(path)          # recarray
    rows = np.asarray([list(row) for row in data], dtype=object)

    # Split features/target
    X_raw = rows[:, :-1]
    y_raw = rows[:, -1]

    # Cast features to float
    X = X_raw.astype(np.float64, copy=False)

    # Decode bytes in y (nominal labels) but keep numeric as-is
    if all(isinstance(v, (bytes, bytearray)) for v in y_raw):
        y = np.array([v.decode("utf-8") for v in y_raw])
    else:
        y = y_raw  # already numeric or str

    return X, y


def plot_results():
    acc = pd.read_csv("accuracy30.csv")
    acc = acc.to_numpy()
    plot, p_vals = plot_critical_difference(acc,
                                    labels=["RandF","RotF","ADA", "GBDT", "CatBoost"],
                             )
    plt.title(f"CD Diagram for Classifier Bake-off on {len(datasets)} UCI Datasets", fontsize=14)
    plt.show()
    plot.savefig("cd_bakeoff30.png")

def run_experiments():

    randf = RandomForestClassifier(n_estimators=100, random_state=42)
    rotf = RotationForestClassifier(n_estimators=100, random_state=42)
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    gbdt = GradientBoostingClassifier(n_estimators=100, random_state=42)
    # xgb = XGBClassifier(n_estimators=100, use_label_encoder=False)
    catb = CatBoostClassifier(iterations=100, verbose=0, random_state=42)
    # tabpfn = TabPFNClassifier()
    cls = [randf, rotf, ada, gbdt, catb] #xgb, tabpfn
    # for name in datasets:
    all_acc = []
    for name in datasets:
        X, y = load_arff_xy("../../data/lab5/"+name+"/"+name+".arff")
        print("Running dataset:", name)
        acc =[]
        for c in cls:
            av = 0
            print("\t\tClassifier:", type(c).__name__)
            for i in range(30):
                trainX, testX, trainy, testy = train_test_split(X,y, test_size=0.3,
                                                             random_state=i, stratify=y)
                c.fit(trainX, trainy)
                a = c.score(testX, testy)
                av+=a
            av = av/30.0
            acc.append(av)
        all_acc.append(acc)
    all_acc = np.array(all_acc)
    cols = ["RandF","RotF","ADA", "GBDT", "CatBoost"]
    np.savetxt("accuracy30.csv", all_acc, delimiter=",", fmt="%.6g",
               header=",".join(cols), comments="")



if __name__ == "__main__":
    # run_experiments()
    plot_results()
