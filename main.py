import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn import svm, datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import random


def plot_results(models, titles, X, y, x_label,y_label,plot_sv=False,):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def SVM_from_features(features,classify):
    C = 10
    df = pd.read_csv("CollectedData_Or.csv")
    temp_X=[]
    for f in features:
        temp_X.append(np.array([float(x) for x in df[f][2:]]).reshape(-1, 1))

    X = np.concatenate(temp_X, axis=1)
    y  =[int(x) for x in df[classify][2:]]
    test_points_indexes = np.random.choice([0, 1], size=len(X), p=[.85, .15])
    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]
    for i,test_point in enumerate(test_points_indexes):
        if test_point:
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])

    models = []
    titles = []

    clf_lin = svm.SVC(C=C)
    clf_lin.set_params(kernel='linear').fit(X_train, y_train)
    y_prediction=[]
    for x in X_test:
        y_prediction.append(clf_lin.predict(x.reshape(1,-1)))
    print (sklearn.metrics.accuracy_score(y_test,y_prediction))
    models.append(clf_lin)
    titles.append("linear kernel")
    """
    clf_poly_2 = svm.SVC(C=C)
    #clf_poly_2.set_params(kernel='poly', degree=2).fit(X, y)
    clf_poly_2.set_params(kernel='poly', degree=2,coef0=1).fit(X, y)
    models.append(clf_poly_2)
    titles.append("poly_2")

    clf_poly_3 = svm.SVC(C=C)
    #clf_poly_3.set_params(kernel='poly', degree=3).fit(X, y)
    clf_poly_3.set_params(kernel='poly', degree=3,coef0=1).fit(X, y)
    models.append(clf_poly_3)
    titles.append("poly_3")


    clf_lin = svm.SVC(C=C)
    clf_lin.set_params(kernel='rbf', gamma=1).fit(X, y)
    models.append(clf_lin)
    titles.append("rbf")

    clf_poly_2 = svm.SVC(C=C)
    #clf_poly_2.set_params(kernel='poly', degree=2).fit(X, y)
    clf_poly_2.set_params(kernel='poly', degree=2,coef0=1).fit(X, y)
    models.append(clf_poly_2)
    titles.append("poly_2")
    """

    plot_results(models, titles, X, y,"angle","distance")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    SVM_from_features(["angle between vertical eye and pupil average","distance between eyelids average"],"gaze shift")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
