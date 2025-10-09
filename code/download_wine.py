import pandas as pd
import matplotlib.pyplot as plt

base = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
red = pd.read_csv(base + "winequality-red.csv", sep=";")
red2 = red.to_numpy()
print(type(red))
print(red.shape)
X = red2[:,0:-1]
y = red["quality"].to_numpy()
print(X.shape, y.shape)
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn import tree

dtc = DecisionTreeClassifier(max_depth=1)
dtr = DecisionTreeRegressor(max_depth=1)
dtc.fit(X,y)
dtr.fit(X,y)
labels = [f"class{i}" for i in range(1, 8)]
tree.plot_tree(
    dtc,
    class_names=[str(c) for c in dtc.classes_],
    label="all",  # show impurity, samples, and value
    filled=True, rounded=True,
    proportion=True,  # <- show class proportions instead of raw counts
    precision=2
)
plt.tight_layout()
plt.show()
print(y)
#
# tree.plot_tree(
#     dtr,
#     filled=True,
#     rounded=True,
# )
# plt.tight_layout()
# plt.show()
#


