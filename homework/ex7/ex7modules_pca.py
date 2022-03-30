import numpy as np
import scipy.linalg as linalg

def pca(X):
    # This function runs principal component analysis on the dataset
    # represented as matrix X. It computes eigenvectors of the covariance matrix of X,
    # and returns the eigenvectors U, the eigenvalues (on diagonal) in S

    # Useful values
    m, n = X.shape

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)

    # ====================== YOUR CODE HERE ======================

    # compute the covariance matrix
    sigma = (1.0/m) * (np.transpose(X)).dot(X)

    # compute the eigenvectors (U) and S
    # from: 
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html#scipy.linalg.svd
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.diagsvd.html#scipy.linalg.diagsvd
    U, S, Vh = linalg.svd(sigma)
    S = linalg.diagsvd(S, len(S), len(S))


    # =============================================================

    return U, S


def projectData(X, U, K):
    # This function computes the projection of the normalized inputs X
    # into reduced dimensional space spanned by the first K columns of U,
    # which are the top K eigenvectors.

    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    # ====================== YOUR CODE HERE ======================
    # get U_reduce for only the desired K
    U_reduce = U[:,:K]

    # get Z - the projections from X onto the space defined by U_reduce
    #	note that this vectorized version performs the projection the instructions
    # 	above but in one operation
    Z = X.dot(U_reduce)

    # =============================================================

    return Z


def recoverData(Z, U, K):
    # This function recovers an approximation the original data
    # that has been reduced to K dimensions from projection.
    # It computes the approximation of the data by projecting back
    # onto the original space using the top K eigenvectors in U.

    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    # ====================== YOUR CODE HERE ======================
    # get U_reduce for only the desired K
    U_reduce = U[:,:K]

    # recover data
    X_rec = Z.dot(np.transpose(U_reduce))


    # =============================================================

    return X_rec