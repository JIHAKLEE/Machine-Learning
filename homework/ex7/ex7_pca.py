#  Principle Component Analysis

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import ex7modules_pca
import ex7utils
import os

if not(os.path.exists('./screenshots')):
    os.makedirs('./screenshots')

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('Visualizing example dataset for PCA.\n')

#  The following command loads the dataset. You should now have the 
#  variable X in your environment
mat = scipy.io.loadmat('ex7data1.mat')
X = np.array(mat["X"])

# interactive graphs
plt.ion()

#  Visualize the example dataset
plt.close()

# kept the scatter() (vs. the plot()) version 
#  because scatter() makes properly circular markers
# plt.plot(X[:, 0], X[:, 1], 'o', markersize=9, markeredgewidth=1, markeredgecolor='b', markerfacecolor='None')
plt.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors='b')
plt.axis([0.5, 6.5, 2, 8])
plt.gca().set_aspect('equal', adjustable='box')
plt.show(block=False)

## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique.
print('Running PCA on example dataset.\n')

#  Before running PCA, it is important to first normalize X
X_norm, mu, _ = ex7utils.featureNormalize(X)

#  Run PCA
U, S = ex7modules_pca.pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
ex7utils.drawLine(mu, mu + 1.5 * S[0,0] * U[:,0].T, c='k', linewidth=2)
ex7utils.drawLine(mu, mu + 1.5 * S[1,1] * U[:,1].T, c='k', linewidth=2)

print('Top eigenvector: \n')
print(' U(:,1) = {:f} {:f} \n'.format(U[0,0], U[1,0]))
print('(you should expect to see -0.707107 -0.707107)')

plt.savefig('screenshots/pca1.png')
input('Program paused. Press enter to continue.')


## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('Dimension reduction on example dataset.\n')

#  Plot the normalized dataset (returned from pca)
plt.close()
plt.scatter(X_norm[:,0], X_norm[:,1], s=75, facecolors='none', edgecolors='b')
plt.axis([-4, 3, -4, 3])
plt.gca().set_aspect('equal', adjustable='box')
plt.show(block=False)

#  Project the data onto K = 1 dimension
K = 1
Z = ex7modules_pca.projectData(X_norm, U, K)
print('Projection of the first example: {}\n'.format(Z[0]))
print('(this value should be about 1.481274)\n')

X_rec  = ex7modules_pca.recoverData(Z, U, K)
print('Approximation of the first example: {:f} {:f}\n'.format(X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)\n')

#  Draw lines connecting the projected points to the original points

plt.scatter(X_rec[:, 0], X_rec[:, 1], s=75, facecolors='none', edgecolors='r')
for i in range(X_norm.shape[0]):
    ex7utils.drawLine(X_norm[i,:], X_rec[i,:], linestyle='--', color='k', linewidth=1)

plt.savefig('screenshots/pca2.png')
input('Program paused. Press enter to continue.')


## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('Loading face dataset.\n')

#  Load Face dataset
mat = scipy.io.loadmat('ex7faces.mat')
X = np.array(mat["X"])

#  Display the first 100 faces in the dataset
plt.close()
ex7utils.displayData(X[:100, :])

plt.savefig('screenshots/faces.png')
input('Program paused. Press enter to continue.')

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('Running PCA on face dataset.\n(this mght take a minute or two ...)\n')

#  Before running PCA, it is important to first normalize X by subtracting 
#  the mean value from each feature
X_norm, _, _ = ex7utils.featureNormalize(X)

#  Run PCA
U, S = ex7modules_pca.pca(X_norm)

#  Visualize the top 36 eigenvectors found
ex7utils.displayData(U[:, :36].T)

plt.savefig('screenshots/facesPCA1.png')

## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('Dimension reduction for face dataset.\n')

K = 100
Z = ex7modules_pca.projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print('{:d} {:d}'.format(Z.shape[0], Z.shape[1]))

input('Program paused. Press enter to continue.')


## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('Visualizing the projected (reduced dimension) faces.\n')

K = 100
X_rec  = ex7modules_pca.recoverData(Z, U, K)

# Display normalized data
plt.close()
plt.subplot(1, 2, 1)
ex7utils.displayData(X_norm[:100,:])
plt.title('Original faces')
plt.gca().set_aspect('equal', adjustable='box')

# Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2)
ex7utils.displayData(X_rec[:100,:])
plt.title('Recovered faces')
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig('screenshots/facesPCA2.png')

input('Program finished. Press enter to exit.')