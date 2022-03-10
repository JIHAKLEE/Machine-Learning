# Machine Learning Online Class - Exercise 1: Linear Regression
# This file contains code that helps you get started on the linear regression exercise.

import numpy as np
import matplotlib.pyplot as plt

from ex1modules import *

# interactive graphs
plt.ion()

## =================== Section 1: Linear regression with one variable ===================
print('[[===== Linear Regression with one variable =====]]')

## =================== Part 1: Loading Data ===================
print('Loading data ...')

# x refers to the population of a city in 10,000s
# y refers to the profit of a food truck in that city in $10,000s
x, y = np.loadtxt('homework\ex1\ex1data1.txt', delimiter=',', unpack=True)
x = np.matrix(x).T
y = np.matrix(y).T

m = len(y) # number of training examples

## =================== Part 2: Gradient descent ===================
print('Running Gradient Descent ...')

X = np.concatenate((np.ones((m, 1)), x), axis=1) # Add a column of ones to x
theta = np.zeros((2, 1)) # initialize fitting parameters

# run gradient descent
[theta, J_history] = gradientDescent(X, y, theta, alpha=0.01, num_iters=1500)

# print theta to screen
print('Theta computed from gradient descent:')
print(theta)

# Plot the data points and the linear function that best fits the given training data set.
plt.plot(x, y, marker='o', linestyle='None',label='training data')
plt.ylabel('Profit in \u20a910,000,000s') # Set the y axis label
plt.xlabel('Population of City in 10,000s') # Set the x axis label
plt.plot(X[:,1], X*theta, '-', label='line fit')
plt.legend()
plt.show()
input('Program paused. Press enter to continue.')
plt.close()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population=35000, we expect a profit of \u20a9{:d}'.format(int(np.round(predict1.item()*10000000))))
predict2 = np.dot([1, 7], theta)
print('For population=70000, we expect a profit of \u20a9{:d}'.format(int(np.round(predict2.item()*10000000))))

## ======== Part 3: Plotting J to iteration index ==============
# plot the cost function value as a function of iteration number
plt.plot(range(0,len(J_history)), J_history, 'r-')
plt.ylabel('J') # Set the y axis label
plt.xlabel('Iteration #') # Set the x axis label
plt.show()
input('Program paused. Press enter to continue.')
plt.close()

## ============= Part 4: Plotting J to parameters =============
print('Visualizing J(theta_0, theta_1) ...')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(0,len(theta0_vals)):
    for j in range(0,len(theta1_vals)):
        t = np.matrix([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i][j] = computeCost(X, y, t)

# Because of the way meshgrids work in the surf command, we need to 
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

fig = plt.figure()
ax = fig.gca(projection='3d')

# Surface plot
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()
input('Program paused. Press enter to continue.')
plt.close()

# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()
input('Program paused. Press enter to continue.')
plt.close()

## =================== Section 2: Linear regression with multiple variable ===================
print('[[===== Linear Regression with two variables =====]]')

## =================== Part 5: Loading Data ===================
print('Loading data ...')

# x1 refers to the size of the house (in square feet)
# x2 refers to the number of bedrooms
# y refers to the price of the house.
x1, x2, y = np.loadtxt('homework\ex1\ex1data2.txt', delimiter=',', unpack=True)
x = np.concatenate((np.matrix(x1), np.matrix(x2))).T
y = np.matrix(y).T

m = len(y) # number of training examples

# Scale features and set them to zero mean
print('Normalizing Features ...')
mu = np.mean(x)
sigma = np.std(x[:,0])
X = np.concatenate( ( np.ones((m, 1)), (x-mu)/sigma ) , axis=1) # Add a column of ones to x

## ================ Part 6: Gradient Descent ================
print('Running gradient descent ...')

# Init Theta and Run Gradient Descent.
# If you have implemented gradientDescent correctly for multivariate case,
# then the following codes should also work.
# If you have implemented gradientDescent correctly ONLY for univariate case,
# then the following codes will NOT work.
theta = np.zeros((3, 1))
[theta, J_history] = gradientDescent(X, y, theta, 0.01, 5000)

# Plot the convergence graph
plt.plot(range(0,len(J_history)), J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()
input('Program paused. Press enter to continue.')

# Display gradient descent's result
print('Theta computed from gradient descent:')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
x_new = np.matrix([1650, 3])
x_new = (x_new-mu)/sigma # Scale features and set them to zero mean
x_new = np.concatenate((np.ones((1, 1)), x_new), axis=1)
price = x_new*theta
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):{:.2f}\n'.format(price.item()))

## ================ Part 7: Normal Equations ================
print('Solving with normal equations...\n')

# Add intercept term to X
X = np.concatenate((np.ones((m, 1)), x), axis=1)

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations:')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
x_new = np.matrix([1650, 3])
x_new = np.concatenate((np.ones((1, 1)), x_new), axis=1)
price = x_new*theta
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equation):{:.2f}\n'.format(price.item()))
