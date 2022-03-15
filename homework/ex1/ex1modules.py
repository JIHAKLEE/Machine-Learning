import numpy as np

def computeCost(X, y, theta):
    # This function computes the cost of using theta as the
    # parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = len(y) # number of training examples

    # ====================== YOUR CODE HERE ===================================
    # TODO: Compute the cost of a particular choice of theta
    # Initialize
    J = 0
    theta = theta.copy() # theta will be initialized at ex1.py
    # numpy explaination
    # numpy.dot() this function returns the dot product of two arrays. For 2-D vectors, it is the equivalent to matrix multiplication
    # np.subtract() this function perform the element wise subtraction
    # np.square() this function perform the element wise square
    # X = np.concatenate((np.ones((m, 1)), x), axis=1) # Add a column of ones to x
    # theta = np.zeros((2, 1)) # initialize fitting parameters    

    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors) 
    J = 1 / (2 * m) * np.sum(sqrErrors)
    # =========================================================================
    return J





def gradientDescent(X, y, theta, alpha, num_iters):
    # This function performs gradient descent to learn theta.
    # It updates theta by taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = len(y) # number of training examples
    n = len(theta) # number of features per training example
    J_history = np.zeros((num_iters, 1))

    for iter in range(0,num_iters):

        # ====================== YOUR CODE HERE ======================
        # TODO: Perform a single gradient step on the parameter vector. 
        #
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * X.transpose().dot(errors)
        theta = theta - sum_delta

        
        # ============================================================
        # Save the cost J in every iteration    
        J_history[iter] = computeCost(X, y, theta)

    return [theta, J_history]





def normalEqn(X, y):
    # This function computes the closed-form solution to linear regression
    # using the normal equations.

    # ====================== YOUR CODE HERE ======================
    # TODO: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    theta = np.zeros((X.shape[1], 1))
    theta = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y))

    # ============================================================
    return theta
