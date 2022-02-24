import numpy as np


def sigmoid(z):
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).

    # =============================================================
    return g

def logRegCost(theta, X, y, regParam=0):
    # When this function is called via op.minimize function, the theta parameter is
    # automatically flattened to common array data type, despite passing the argument as np.matrix type.
    # So, the following couple of lines of code are added to deal with this issue.
    if type(theta)==np.ndarray:
        theta = np.matrix(theta).T
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #

    # =============================================================

    return J

def logRegGrad(theta, X, y, regParam=0):
    # When this function is called via op.minimize function, the theta parameter is
    # automatically flattened to common array data type, despite passing the argument as np.matrix type.
    # So, the following couple of lines of code are added to deal with this issue.
    if type(theta)==np.ndarray:
        theta = np.matrix(theta).T
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #

    # =============================================================

    return grad

def predict(theta, X):
    # ====================== YOUR CODE HERE ======================
    # Instructions: Predict whether the label is 0 or 1 using learned logistic 
    #               regression parameters theta. Use threshold at 0.5.
    #

    # =============================================================
    return p


def mapFeature(X1, X2):
    # This function maps the two input features to quadratic features 
    # used in the regularization exercise. It returns a new feature array
    # with more features, comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    # Inputs X1, X2 must be the same size

    degree = 6
    if np.isscalar(X1):
       X1 = np.matrix(X1)
    if np.isscalar(X2):
       X2 = np.matrix(X2)
    out = np.ones(X1.shape)
    for i in range(1,degree+1):
        for j in range(0,i+1):
            out = np.concatenate((out, np.multiply(np.power(X1,i-j),np.power(X2,j))), axis=1)
    return out
