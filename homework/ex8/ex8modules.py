import numpy as np

def estimateGaussian(X):
    # This function estimates the parameters of a Gaussian distribution using the data in X.
    # The input X is the dataset with each n-dimensional data point in one row.
    # The output is an n-dimensional vector mu, the feature-wise mean values of the data set
    # and an n-dimensional vector sigma2, the feature-wise variance values of the data set.
    # In other words, mu[i] should contain the mean of the data for the i-th feature and
    # sigma2[i] should contain variance of the data of the i-th feature.

    # Useful variables
    m, n = X.shape

    # You should return these values correctly
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))

    # ====================== YOUR CODE HERE ======================
   
    mu = np.mean(X, axis=0)
    sigma2 = np.var(X, axis=0)

    # =============================================================

    return mu, sigma2

def selectThreshold(yval, pval):
    # This function finds the best threshold (bestEpsilon) to use for selecting outliers
    # based on the results from a validation set (pval) and the ground truth (yval).
    # It computes the F1 score of choosing epsilon as the threshold and places
    # the value in F1. The code at the end of the loop should compare the F1 score
    # for the given choice of epsilon and set it to be the best epsilon if
    # it is better than the current choice of epsilon.
    # Note: You can use predictions = (pval < epsilon) to get a binary vector
    #       of 0's and 1's of the outlier predictions

    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval), stepsize):
        # ====================== YOUR CODE HERE ======================

        predictions = np.less(pval, epsilon)
        tp = np.sum(np.logical_and(predictions, yval))
        fp = np.sum(np.logical_and(predictions, yval == 0))
        fn = np.sum(np.logical_and(np.logical_not(predictions), yval == 1))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = (2 * precision * recall) / (precision + recall)

        # =============================================================

        if F1 > bestF1:
           bestF1 = F1
           bestEpsilon = epsilon
    return bestEpsilon, bestF1