import numpy as np
import ex6utils

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
    sim = np.exp((-1) / (2 * sigma**2) * sum((x1 - x2) * (x1 - x2).T))
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
    #               predictions = model.predict(ex6utils.gaussianKernelGramMatrix(Xval, X))
    #               will return the predictions on the cross validation set.
    #
    #  Note: You can compute the prediction error using 
    #        mean(double(predictions ~= yval))
    #
    ### determining best C and sigma

    # only uncomment if similar lines are uncommented on svmTrain.py
    # yval = yval.astype("int32")
    # yval[yval==0] = -1

    # vector with all predictions from SVM
    x1 = [1, 2, 1]
    x2 = [0, 4, -1]
    predictionErrors = np.zeros((64,3))
    predictionsCounter = 0

    for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for C in [0.01, 0.03, 0.2, 0.3, 1, 3, 10, 30]:
            print("trying sigma={:.2f}, C={:.2f}".format(sigma, C))

            model = ex6utils.svmTrain(X, y, C, "gaussian", tol=1e-3, max_passes=-1, sigma = sigma)
            predictions = model.predict(ex6utils.gaussianKernelGramMatrix(Xval, X))
            predictionErrors[predictionsCounter, 0] = np.mean((predictions != yval).astype(int))

            predictionErrors[predictionsCounter, 1] = sigma
            predictionErrors[predictionsCounter, 2] = C
            predictionsCounter += 1
    print(predictionErrors)

    row = predictionErrors.argmin(axis=0)
    m = np.zeros(row.shape)
    for i in range(len(m)):
        m[i] = predictionErrors[row[i]][i]
    
    print(predictionErrors[row[0], 1])
    print(predictionErrors[row[0], 2])

    sigma = predictionErrors[row[0], 1]
    C = predictionErrors[row[0], 2]

    # ==================================================================

    return C, sigma
