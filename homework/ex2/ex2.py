## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  This file contains code that helps you get started on the first part of logistic regression exercise. 

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

from ex2modules import *

plt.ion()

## Load Data
# x1 refers to Exam 1 score
# x2 refers to Exam 2 score
# y refers to whether or not a given student is admitted.

data = np.loadtxt('homework\ex2\ex2data1.txt',delimiter=',')
x = np.matrix(data[:, 0:2])
y = np.matrix(data[:, 2:3])


## ==================== Part 1: Plotting ====================
#  We start the exercise by plotting the positive and negative examples on a 2D plot.

# Find Indices of Positive and Negative Examples
pos = np.where(y==1)
neg = np.where(y==0)
# Plot Examples
plt.scatter([x[pos,0]], [x[pos,1]], marker='o', c='blue', s=25, label='Admitted')
plt.scatter([x[neg,0]], [x[neg,1]], marker='x', c='red', s=25, label='Not admitted')

# Put some Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

plt.legend(scatterpoints=1)
plt.show()

## ============ Part 2: Compute Initial Cost and Gradient ============

#  Setup the data matrix appropriately.
[m, n] = x.shape
X = np.concatenate((np.ones((m, 1)), x), axis=1) # Add intercept term to x and X_test
initial_theta = np.matrix(np.zeros((n + 1, 1))) # Initialize fitting parameters


cost = logRegCost(initial_theta, X, y)
grad = logRegGrad(initial_theta, X, y)

print('\nCost at initial theta (np.zeros): {}'.format(cost.item()))
print('Gradient at initial theta (np.zeros):')
print(grad)

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc/fmincg)
#  to find the optimal parameters theta.

Result = op.minimize(fun=logRegCost, x0=initial_theta, args=(X, y), method='TNC',jac=logRegGrad)
theta = Result.x

# Print theta to screen
print('\nCost at optimal theta: {}'.format(Result.fun))
print('The value of optimal theta:')
print(np.matrix(theta).T)

if n <= 2:
    # Only need 2 points to define a line, so choose two endpoints
    plot_x = np.matrix([np.min(X[:,1])-2, np.max(X[:,1])+2]).T

    # Calculate the decision boundary line
    plot_y = np.multiply(-1.0/theta[2],(np.multiply(theta[1],plot_x) + theta[0]))

    # Plot, and adjust axes for better viewing
    plt.plot(plot_x, plot_y, label='Decision boundary')
    
    # Legend, specific for the exercise
    plt.legend()
    plt.axis([30, 100, 30, 100])
else:
    # Here is the grid range
    u = linspace(-1, 1.5, 50)
    v = linspace(-1, 1.5, 50)

    z = np.zeros(length(u), length(v))
    # Evaluate z = theta*x over the grid
    for i in range(0,length(u)):
        for j in range(0,length(v)):
            z[i,j] = mapFeature(u[i], v[j])*np.matrix(theta).T
    z = z.T # important to transpose z before calling contour

    # Plot z = 0
    # Notice you need to specify the range [0, 0]
    plt.contour(u, v, z, [0, 0], linewidth=2)

# Put some labels 
plt.legend()
plt.show()
input('Program paused. Press Enter to continue.')

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2 

prob = sigmoid([1, 45, 85] * np.matrix(theta).T)
print('\nFor a student with scores 45 and 85, we predict an admission probability of {}'.format(prob.item()))

# Compute accuracy on our training set
prediction = predict(np.matrix(theta).T, X)
print('Training Accuracy: {:.2f}%\n'.format(np.mean(prediction == y) * 100))
