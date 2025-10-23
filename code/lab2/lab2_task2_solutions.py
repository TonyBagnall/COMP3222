
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn import tree
"""
Lab 2 Task 2: decision tree classification in scikit-learn

this task is about using these classifiers and visualising the output using plot_tree. 
The exercise is similar to those in Chapter 6 of HOML (page 195 onwards), but using 
our play golf example.

The data is loaded from ../../data/lab_2/playgolf.csv into a pandas DataFrame. This 
is the easiest way to load CSV data with mixed data types (strings and numbers).
The target variable is in column 4 (PlayGolf: Yes/No),

We split the data into features X (columns 0 to 3) and labels y (column 4). The rest 
is self explanatory.

"""
if __name__ == "__main__":    # Example usage
    # print(f"Gini Impurity: {impurity}")
    data = pd.read_csv('../../data/lab_2/playgolf.csv')
    print(data)
    y = data.iloc[:, 4].to_numpy()
    print(y)
    X = data.iloc[:, 0:4].to_numpy()
    print(X.shape)
    outlook = X[:, 0]
    print(outlook)
    print(outlook.shape)
    ohe = OneHotEncoder(sparse_output=False)
    outlook_encoded = ohe.fit_transform(outlook.reshape(-1, 1))
    print(outlook_encoded)
    print(outlook_encoded.shape)
    dt = DecisionTreeClassifier()
    dt = ExtraTreeClassifier()
    X_rest = X[:, 1:]  # drop original outlook
    X_combined = np.hstack((outlook_encoded, X_rest))
    dt.fit(X_combined,y)
    dt = HistGradientBoostingClassifier()
    dt.fit(X_combined,y)

    # Build feature names for the transformed matrix
    ohe_names = ohe.get_feature_names_out([data.columns[0]])  # ["outlook_..."]
    rest_names = list(data.columns[1:4])  # original cols 1..3
    feature_names = list(ohe_names) + rest_names
    class_names = ["No", "Yes"]
    # Plot
    plt.figure(figsize=(10, 6))
    tree.plot_tree(
        dt,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
    )
    plt.tight_layout()
    plt.show()