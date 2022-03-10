import numpy as np
import scipy.optimize as op

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

def trainOneVsAll(X, y, num_labels, regParam=0):
    # This function trains multiple logistic regression classifiers and returns
    # the classifiers in a matrix all_theta, where the i-th row of all_theta corresponds
    # to the classifier for label i.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to train num_labels
    #               logistic regression classifiers with regularization parameter.
    #
    # Hint: You can implement this using scipy.optimize.minimize() with a loop:
    #
    #    for c in range(1,num_labels+1):
    #        Result = op.minimize(fun=logRegCost, x0=initial_theta, args=(X, (y==c) ,regParam), method='TNC',jac=logRegGrad)
    #
    # Here, the `Result` object has optimized theta values for given label value c,
    # stored in member variable x. Your task is to pack those label-wise values to
    # matrix all_theta.

    # =========================================================================
    return all_theta


def predictOneVsAll(all_theta, X):
    # This function predicts the label for a trained one-vs-all classifier.
    # The function will return a vector of predictions for each example in the matrix X,
    # where X contains the examples in rows.
    # all_theta is a matrix where the i-th row is a trained logistic regression theta vector for the i-th class.

    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly 
    predictions = np.zeros((m, 1))

    # Add ones to the X data matrix
    X = np.concatenate( ( np.ones((m,1)), X ), axis = 1 )


    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters (one-vs-all).
    #               You should set 'predictions' to a vector of predictions,
    #               ranging from 1 to num_labels.
    #               (e.g., predictions = [1; 3; 1; 2] predicts classes 1, 3, 1, 2 for 4 examples)
    #    

    # =========================================================================

    return predictions
