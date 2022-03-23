import numpy as np
import ex6utils

import numpy as np

def gaussianKernel(x1, x2, sigma=0.1):
    #RBFKERNEL returns a radial basis function kernel between x1 and x2
    #   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    #   and returns the value in sim

    # Ensure that x1 and x2 are column vectors
    x1 = x1.flatten()
    x2 = x2.flatten()

    # You need to return the following variables correctly.
    sim = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the similarity between x1
    #               and x2 computed using a Gaussian kernel with bandwidth
    #               sigma
    #
    #

    # =============================================================
        
    return sim

def dataset3Params(X, y, Xval, yval):
    # This function returns your choice of C and sigma. You should complete this
    # function to return the optimal C and sigma based on a cross-validation set.
    #

    # You need to return the following variables correctly.
    sigma = 0.3
    C = 1

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You can use svmPredict to predict the labels on the cross
    #               validation set. For example, 
    #                   predictions = svmPredict(model, Xval)
    #               will return the predictions on the cross validation set.
    #
    #  Note: You can compute the prediction error using 
    #        mean(double(predictions ~= yval))
    #

    # ==================================================================

    return C, sigma
