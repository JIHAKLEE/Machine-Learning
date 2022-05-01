#  Anomaly Detection and Collaborative Filtering

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import ex8modules
import ex8utils

import os
if not(os.path.exists('./screenshots')):
    os.makedirs('./screenshots')

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easy to
#  visualize.
#
#  Our example case consists of 2 network server statistics across
#  several machines: the latency and throughput of each machine.
#  This exercise will help us find possibly faulty (or very fast) machines.
#

print('Visualizing example dataset for outlier detection.\n');

#  The following command loads the dataset. You should now have the
#  variables X, Xval, yval in your environment
mat = scipy.io.loadmat('ex8data1.mat')
X = mat["X"]
Xval = mat["Xval"]
yval = mat["yval"].flatten()

#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bx', markersize=10, markeredgewidth=1)
plt.axis([0,30,0,30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show(block=False)


## ================== Part 2: Estimate the dataset statistics ===================
#  For this exercise, we assume a Gaussian distribution for the dataset.
#
#  We first estimate the parameters of our assumed Gaussian distribution, 
#  then compute the probabilities for each of the points and then visualize 
#  both the overall distribution and where each of the points falls in 
#  terms of that distribution.
#
print('Visualizing Gaussian fit.\n')

#  Estimate my and sigma2
mu, sigma2 = ex8modules.estimateGaussian(X)

#  Returns the density of the multivariate normal at each data point (row) 
#  of X
p = ex8utils.multivariateGaussian(X, mu, sigma2)

#  Visualize the fit
plt.ion()
X1,X2 = np.meshgrid(np.arange(0, 35.1, 0.5), np.arange(0, 35.1, 0.5))
Z = ex8utils.multivariateGaussian(np.column_stack((X1.reshape(X1.size, order='F'), X2.reshape(X2.size, order='F'))), mu, sigma2)
Z = Z.reshape(X1.shape, order='F')

plt.plot(X[:, 0], X[:, 1],'bx', markersize=13, markeredgewidth=1)
# plt.scatter(X[:, 0], X[:, 1], s=150, c='b', marker='x', linewidths=1)

# Do not plot if there are infinities
if (np.sum(np.isinf(Z)) == 0):
    plt.contour(X1, X2, Z, np.power(10,(np.arange(-20, 0.1, 3)).T))
    
plt.show(block=False)

## ================== Part 3: Find Outliers ===================
#  Now you will find a good epsilon threshold using a cross-validation set
#  probabilities given the estimated Gaussian distribution
# 

pval = ex8utils.multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = ex8modules.selectThreshold(yval, pval)
print('Best epsilon found using cross-validation: {:e} (this value should be about 8.99e-05)'.format(epsilon))
print('Best F1 on Cross Validation Set:  {:f}'.format(F1))

# Find the outliers in the training set and plot the
outliers = p < epsilon

# interactive graphs
plt.ion()

#  Draw a red circle around those outliers
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', linewidth=2, markersize=18, fillstyle='none', markeredgewidth=1)
plt.show(block=False)


## ================== Part 4: Multidimensional Outliers ===================
#  We will now use the code from the previous part and apply it to a 
#  harder problem in which more features describe each datapoint and only 
#  some features indicate whether a point is an outlier.
#

#  Loads the second dataset. You should now have the
#  variables X, Xval, yval in your environment
mat = scipy.io.loadmat('ex8data2.mat')
X = mat["X"]
Xval = mat["Xval"]
yval = mat["yval"].flatten()

#  Apply the same steps to the larger dataset
mu, sigma2 = ex8modules.estimateGaussian(X)

#  Training set 
p = ex8utils.multivariateGaussian(X, mu, sigma2)

#  Cross-validation set
pval = ex8utils.multivariateGaussian(Xval, mu, sigma2)

#  Find the best threshold
epsilon, F1 = ex8modules.selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: {:e} (this value should be about 1.38e-18)'.format(epsilon))
print('Best F1 on Cross Validation Set:  {:f}'.format(F1))
print('# Outliers found: {:d}'.format(np.sum((p < epsilon).astype(int))))

plt.savefig('screenshots/ex8.png')
input('Program finished. Press enter to exit.')
