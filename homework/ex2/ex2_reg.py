## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  This file contains code that helps you get started on the second part of logistic regression exercise. 

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

from ex2modules import *

plt.ion()

## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = np.loadtxt('ex2data2.txt',delimiter=',')
x = np.matrix(data[:, 0:2])
y = np.matrix(data[:, 2:3])

# Find Indices of Positive and Negative Examples
pos = np.where(y==1)
neg = np.where(y==0)
# Plot Examples
plt.scatter([x[pos,0]], [x[pos,1]], marker='o', c='blue', s=25, label='Accepted')
plt.scatter([x[neg,0]], [x[neg,1]], marker='x', c='red', s=25, label='Rejected')

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

plt.legend(scatterpoints=1)

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic 
#  regression to classify the data points. 
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(x[:,0], x[:,1])

[m, n] = X.shape

# Initialize fitting parameters
initial_theta = np.matrix(np.zeros((n, 1)))

# Set regularization parameter lambda to 1
regParam = 1

# Compute and display initial cost and gradient for regularized logistic regression
cost = logRegCost(initial_theta, X, y, regParam)
grad = logRegGrad(initial_theta, X, y, regParam)

print('\nCost at initial theta (np.zeros): {}'.format(cost.item()))

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and 
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

Result = op.minimize(fun=logRegCost, x0=initial_theta, args=(X, y,regParam), method='TNC',jac=logRegGrad)
theta = Result.x

# Plot Boundary
if n <= 2:
    # Only need 2 points to define a line, so choose two endpoints
    plot_x = np.matrix([np.asscalar(min(X[:,1])-2),  np.asscalar(max(X[:,1])+2)]).T

    # Calculate the decision boundary line
    plot_y = np.multiply(-1.0/theta[2],(np.multiply(theta[1],plot_x) + theta[0]))

    # Plot, and adjust axes for better viewing
    plt.plot(plot_x, plot_y, label='Decision boundary')
    
    # Legend, specific for the exercise
    plt.legend()
    plt.axis([30, 100, 30, 100])
else:
    # Here is the grid range
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(u), len(v)))
    # Evaluate z = theta*x over the grid
    for i in range(0,len(u)):
        for j in range(0,len(v)):
            z[i,j] = mapFeature(u[i], v[j])*np.matrix(theta).T
    z = z.T # important to transpose z before calling contour

    # Plot z = 0
    # Notice you need to specify the range [0, 0]
    plt.contour(u, v, z, np.linspace(-2, 3, 20), linewidth=2)


plt.title('lambda = {}'.format(regParam))
plt.show()
input('Program paused. Press Enter to continue.')

# Compute accuracy on our training set
prediction = predict(np.matrix(theta).T, X)
print('Training Accuracy: {:.2f}%\n'.format(np.mean(prediction == y) * 100))


