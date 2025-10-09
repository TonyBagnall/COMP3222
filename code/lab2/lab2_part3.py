from aeon.datasets import load_gunpoint, load_from_ts_file,load_italy_power_demand
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np


def load_data(split="train"):
    X, y = load_italy_power_demand(return_type="numpy2D")
    # X, y = load_from_ts_file("../../data/lab_2/gunpoint_TRAIN",
    #                          return_type="numpy2D")
    y = y.astype(np.int64)
    print(y)
    y = y-1
    return X, y


def plot_tree(dt):
    class_names =["Normal", "Anomaly"]
    feature_names = [f"t{i}" for i in range(1, 24)]
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


from sklearn.tree import _tree

def plot_tree_2d(
    clf,
    X,
    y=None,
    feature_names=None,
    class_names=None,
    ax=None,
    grid_points=400,
    margin=0.5,
    region_alpha=0.25,
    show_data=True,
    show_splits=True,
    annotate_depth=False,
    linewidth=2,
):
    """
    Plot decision regions and split lines for a fitted scikit-learn DecisionTree on 2D data.

    Parameters
    ----------
    clf : sklearn.tree.DecisionTreeClassifier or Regressor (fitted)
        Must expose .predict and .tree_.
    X : array-like of shape (n_samples, 2)
        Two input features.
    y : array-like of shape (n_samples,), optional
        Class labels for colouring markers. If None, points are unlabelled.
    feature_names : list/tuple of length 2, optional
        Axis labels.
    class_names : list-like of length n_classes, optional
        Legend labels for classes.
    ax : matplotlib Axes, optional
        Axes to draw on. If None, a new figure/axes is created.
    grid_points : int, default=400
        Resolution of the background decision map.
    margin : float, default=0.5
        Padding around min/max of each axis.
    region_alpha : float, default=0.25
        Transparency of the decision regions.
    show_data : bool, default=True
        Scatter the training points.
    show_splits : bool, default=True
        Draw split lines from the trained tree (solid for depth 0, dashed for 1,
        dotted for 2, dash-dot for 3+).
    annotate_depth : bool, default=False
        Add rough depth annotations on the plot.
    linewidth : float, default=2
        Line width for split lines.

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be of shape (n_samples, 2).")

    # Create axes
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    # Bounds
    x0_min, x0_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    x1_min, x1_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    # Background decision regions
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_points),
        np.linspace(x1_min, x1_max, grid_points),
    )
    Z = clf.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)
    ax.contourf(xx0, xx1, Z, alpha=region_alpha)

    # Data points
    if show_data:
        if y is None:
            ax.scatter(X[:, 0], X[:, 1], marker="o", edgecolor="k")
        else:
            y = np.asarray(y)
            classes = np.unique(y)
            markers = ["o", "s", "^", "v", "P", "X", "D", "*", "<", ">", "h"]
            for i, cls in enumerate(classes):
                lab = (
                    class_names[int(cls)]
                    if (class_names is not None and int(cls) < len(class_names))
                    else str(cls)
                )
                ax.scatter(
                    X[y == cls, 0],
                    X[y == cls, 1],
                    marker=markers[i % len(markers)],
                    label=lab,
                )
            ax.legend(loc="best", framealpha=0.95)

    # Axis labels
    if feature_names is not None and len(feature_names) == 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    # Split lines from the trained tree
    if show_splits and hasattr(clf, "tree_"):
        tree = clf.tree_
        feat = tree.feature
        thr = tree.threshold
        left = tree.children_left
        right = tree.children_right
        styles = ["-", "--", ":", "-."]

        def draw(node, depth, xb, yb):
            f = feat[node]
            t = thr[node]
            if f == _tree.TREE_UNDEFINED:
                return
            style = styles[min(depth, 3)]
            if f == 0:  # vertical split (feature 0)
                x = float(t)
                x = min(max(x, xb[0]), xb[1])
                ax.plot([x, x], [yb[0], yb[1]], linestyle=style, linewidth=linewidth)
                draw(left[node], depth + 1, (xb[0], min(x, xb[1])), yb)
                draw(right[node], depth + 1, (max(x, xb[0]), xb[1]), yb)
            elif f == 1:  # horizontal split (feature 1)
                yline = float(t)
                yline = min(max(yline, yb[0]), yb[1])
                ax.plot([xb[0], xb[1]], [yline, yline], linestyle=style, linewidth=linewidth)
                draw(left[node], depth + 1, xb, (yb[0], min(yline, yb[1])))
                draw(right[node], depth + 1, xb, (max(yline, yb[0]), yb[1]))

        draw(0, 0, (x0_min, x0_max), (x1_min, x1_max))

        if annotate_depth:
            ax.text(x0_min + 0.35, x1_min + 0.3, "Depth=0")
            ax.text((x0_min + x0_max) / 2.6, (x1_min + x1_max) / 2.1, "Depth=1")
            ax.text(x0_max - 1.8, (x1_min + x1_max) / 2.3, "(Depth=2)")

    ax.set_xlim(x0_min, x0_max)
    ax.set_ylim(x1_min, x1_max)
    ax.set_title("Decision tree decision boundaries")
    plt.tight_layout()
    plt.show()
    return ax



if __name__ == "__main__":    # Example usage
    X, y = load_data()
    print(X.shape, y.shape)
    print(X.shape, y.shape)
    print("Unique = ", set(y))
    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    plot_tree(dt)
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    iris = load_iris(as_frame=True)
    X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y_iris = iris.target
    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    tree_clf.fit(X_iris, y_iris)
    print(y_iris)
    a=plot_tree_2d(tree_clf,X_iris, y_iris)


