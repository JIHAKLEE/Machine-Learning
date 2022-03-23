import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import ex6modules

# from @lejlot http://stackoverflow.com/a/26962861/583834
def gaussianKernelGramMatrix(X1, X2, K_function=ex6modules.gaussianKernel, sigma=0.1):
    """(Pre)calculates Gram Matrix K"""

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = K_function(x1, x2, sigma)
    return gram_matrix

def plotData(X, y):
    # This function plots the data points with + for the positive examples
    # and o for the negative examples. X is assumed to be a Mx2 matrix.
    #
    # Note: This was slightly modified such that it expects y = 1 or y = 0

    # Find Indices of Positive and Negative Examples
    y = y.flatten()
    pos = y==1
    neg = y==0

    # Plot Examples
    plt.plot(X[:,0][pos], X[:,1][pos], "k+", markersize=10)
    plt.plot(X[:,0][neg], X[:,1][neg], "yo", markersize=10)

    # alternatives
    #
    # plt.scatter(*zip(*X[pos]), c="k", marker='+', s=100)
    # plt.scatter(*zip(*X[neg]), c="y", marker='o', s=100)
    #
    # plt.plot(X[:,0][pos], X[:,1][pos], 'k+', X[:,0][neg], X[:,1][neg], 'yo', markersize=10)

    plt.show(block=False) 

def visualizeBoundary(X, y, model, varargin=0):
    #VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
    #   boundary learned by the SVM and overlays the data on it

    # Plot the training data on top of the boundary
    plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
    x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
       this_X = np.column_stack((X1[:, i], X2[:, i]))
       vals[:, i] = model.predict(gaussianKernelGramMatrix(this_X, X))

    # Plot the SVM boundary
    plt.contour(X1, X2, vals, colors="blue", levels=[0,0])
    plt.show(block=False)


def visualizeBoundaryLinear(X, y, model):
    #VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
    #SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
    #   learned by the SVM and overlays the data on it

    # plot decision boundary
    # right assignments from http://stackoverflow.com/a/22356267/583834
    w = model.coef_[0]
    b = model.intercept_[0]
    xp = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    yp = - (w[0] * xp + b) / w[1]

    plt.plot(xp, yp, 'b-')

    # plot training data
    plotData(X, y)


def svmTrain(X, y, C, kernelFunction, tol=1e-3, max_passes=-1, sigma=0.1):
    """Trains an SVM classifier"""

    y = y.flatten() # prevents warning

    # alternative to emulate mapping of 0 -> -1 in svmTrain.m
    #  but results are identical without it
    # also need to cast from unsigned int to regular int
    # otherwise, contour() in visualizeBoundary.py doesn't work as expected
    # y = y.astype("int32")
    # y[y==0] = -1

    if kernelFunction == "gaussian":
        clf = svm.SVC(C = C, kernel="precomputed", tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(gaussianKernelGramMatrix(X,X, sigma=sigma), y)

    # elif kernelFunction == "linear":
    #     clf = svm.SVC(C = C, kernel="precomputed", tol=tol, max_iter=max_passes, verbose=2)
    #     return clf.fit(np.dot(X,X.T).T, y)

    else: # works with "linear", "rbf"
        clf = svm.SVC(C = C, kernel=kernelFunction, tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(X, y)
