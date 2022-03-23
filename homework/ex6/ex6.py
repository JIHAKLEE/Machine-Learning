import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import ex6utils
import ex6modules


## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

print('Loading and Visualizing Data ...')

# Load from ex6data1: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data1.mat')
X = mat["X"]
y = mat["y"]

ex6utils.plotData(X, y)


## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#

print('Training Linear SVM ...')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
model = ex6utils.svmTrain(X, y, C, "linear", 1e-3, 20)
plt.close()
ex6utils.visualizeBoundaryLinear(X, y, model)

input('Program paused. Press enter to continue.')

## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('Evaluating the Gaussian Kernel ...')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = ex6modules.gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma=2 : {}\n(this value should be about 0.324652)\n'.format(sim))

input('Program paused. Press enter to continue.')

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...')

# Load from ex6data2: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data2.mat')
X = mat["X"]
y = mat["y"]

# Plot training data
plt.close()
ex6utils.plotData(X, y)

input('Program paused. Press enter to continue.')

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.
# 
print('Training SVM with Gaussian Kernel (plotting the boundary may take 1 to 2 minutes) ...')

# SVM Parameters
C = 1
sigma = 0.1

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.
model = ex6utils.svmTrain(X, y, C, "gaussian")

plt.close()
ex6utils.visualizeBoundary(X, y, model)

input('Program paused. Press enter to continue.')

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...')

# Load from ex6data3: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data3.mat')
X = mat["X"]
y = mat["y"]
Xval = mat["Xval"]
yval = mat["yval"]

# Plot training data
plt.close()
ex6utils.plotData(X, y)

input('Program paused. Press enter to continue.')

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
# 

# Try different SVM Parameters here
C, sigma = ex6modules.dataset3Params(X, y, Xval, yval)

# train model on training corpus with current sigma and C
model = ex6utils.svmTrain(X, y, C, "gaussian", sigma=sigma)
plt.close()
ex6utils.visualizeBoundary(X, y, model)

input('Program paused. Press enter to continue.')

