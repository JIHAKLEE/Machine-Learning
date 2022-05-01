import numpy as np
import scipy.linalg as linalg

import ex8modules_cofi

def multivariateGaussian(X, mu, sigma2):
    # This function computes the probability 
    # density function of the examples X under the multivariate gaussian 
    # distribution with parameters mu and sigma2. If sigma2 is a matrix, it is
    # treated as the covariance matrix. If sigma2 is a vector, it is treated
    # as the \sigma^2 values of the variances in each dimension (a diagonal
    # covariance matrix)
    #

    k = len(mu)

    # turns 1D array into 2D array
    if sigma2.ndim == 1:
        sigma2 = np.reshape(sigma2, (-1,sigma2.shape[0]))

    if sigma2.shape[1] == 1 or sigma2.shape[0] == 1:
        sigma2 = linalg.diagsvd(sigma2.flatten(), len(sigma2.flatten()), len(sigma2.flatten()))

    # mu is unrolled (and transposed) here
    X = X - mu.reshape(mu.size, order='F').T

    p = np.dot(np.power(2 * np.pi, - k / 2.0), np.power(np.linalg.det(sigma2), -0.5) ) * \
        np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(sigma2)) * X, axis=1))

    return p


def computeNumericalGradient(J, theta):
    # This function computes the numerical gradient of the function J around theta.
    # It does so using "finite differences" and gives us a numerical estimate of the gradient.
    # Notes: The following code implements numerical gradient checking, and 
    #        returns the numerical gradient.It sets numgrad(i) to (a numerical 
    #        approximation of) the partial derivative of J with respect to the 
    #        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
    #        be the (approximately) the partial derivative of J with respect 
    #        to theta(i).)
    #                

    numgrad = np.zeros( theta.shape )
    perturb = np.zeros( theta.shape )
    e = 1e-4

    for p in range(theta.size):
        # Set perturbation vector
        perturb.reshape(perturb.size, order="F")[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad.reshape(numgrad.size, order="F")[p] = (loss2 - loss1) / (2*e)
        perturb.reshape(perturb.size, order="F")[p] = 0

    return numgrad


def checkCostFunction(lambda_var=0):
    # This function creates a small collaborative filering problem to check the cost function and its backpropagation gradients.
    # It will output the analytical gradients produced by your backprop code and
    # the numerical gradients (computed using computeNumericalGradient).
    # These two gradient computations should result in very similar values.
    #

    ## Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    ## Run Gradient Checking
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

    # Short hand for cost function
    def costFunc(p):
        return ex8modules_cofi.cofiCostFunc(p, Y, R, num_users, num_movies, num_features, lambda_var)

    numgrad = computeNumericalGradient(costFunc, params)
    cost, grad = ex8modules_cofi.cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_var)

    print(np.column_stack((numgrad, grad)))
    print('If your backpropagation implementation is correct, the above two columns you get should be very similar.')

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('Relative Difference: {:e} (this value should be less than 1e-9)'.format(diff))

