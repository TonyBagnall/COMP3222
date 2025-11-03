from sklearn.ensemble import RandomForestClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.datasets import load_classification, load_gunpoint
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix, log_loss
from sklearn.metrics import roc_auc_score, precision_score, matthews_corrcoef, f1_score
from sklearn import metrics as skm
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


def gunpoint_example():
    randf = RandomForestClassifier(n_estimators=100, random_state=42)
    rotf = RotationForestClassifier(n_estimators=100, random_state=42)
    trainX, trainy = load_gunpoint( split="train", return_type="numpy2D")
    testX, testy = load_gunpoint(split="test", return_type="numpy2D")
    labels, counts = np.unique(testy, return_counts=True)
    print(counts)
    print(labels)
    randf.fit(trainX, trainy)
    rotf.fit(trainX, trainy)
    print(trainy)
    pred1 = randf.predict(testX)
    pred2 = rotf.predict(testX)
    prob1 = randf.predict_proba(testX)
    prob2 = rotf.predict_proba(testX)
    auc1 = skm.roc_auc_score(testy, prob1[:,1])
    auc2 = skm.roc_auc_score(testy, prob2[:,1])
    print(auc1, auc2)

    a1 = accuracy_score(testy, pred1)
    a2 = accuracy_score(testy, pred2)
    print(a1, a2)
    cm1 = confusion_matrix(testy, pred1)
    cm2 = confusion_matrix(testy, pred2)
    print(cm1.T)
    print(cm2.T)
    ba1 = balanced_accuracy_score(testy, pred1)
    ba2 = balanced_accuracy_score(testy, pred2)
    print(ba1, ba2)
    nll1 = log_loss(testy, prob1)
    nll2 = log_loss(testy, prob2)
    print(nll1, nll2)
    p1=precision_score(testy, pred1, pos_label = '2')
    p2=precision_score(testy, pred2, pos_label = '2')
    print("precision",p1,p2)
    f1a1=f1_score(testy, pred1,pos_label = '2')
    f1a2=f1_score(testy, pred2,pos_label = '2')
    print("f1",f1a1, f1a2)
    m1=matthews_corrcoef(testy, pred1)
    m2=matthews_corrcoef(testy, pred2)
    print("matthews correlation",m1,m2)
    # y_true can be strings, e.g. '1'/'2'; y_score must be the score/prob for the POSITIVE class
    # Example: positive class is '2'
    RocCurveDisplay.from_predictions(testy, prob1[:,1], pos_label='2')
    plt.title(f"ROC curve (AUC = {auc1:.3f})")
    plt.show()

if __name__ == "__main__":
    gunpoint_example()

