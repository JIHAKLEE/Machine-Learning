#  Anomaly Detection and Collaborative Filtering

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import ex8modules_cofi
import ex8utils

## =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.
#  
print('Loading movie ratings dataset.\n')

#  Load data
mat = scipy.io.loadmat('homework\ex8\ex8_movies.mat')
Y = mat["Y"]
R = mat["R"]

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): {:f} / 5\n'.format(np.mean(Y[0, R[0, :]==1])))

#  We can "visualize" the ratings matrix by plotting it with imagesc
# need aspect='auto' for a ~16:9 (vs almost vertical) aspect
plt.imshow(Y, aspect='auto') 
plt.ylabel('Movies')
plt.xlabel('Users')
plt.show(block=False)

## ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that.

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
mat = scipy.io.loadmat('homework\ex8\ex8_movieParams.mat')
X = mat["X"]
Theta = mat["Theta"]
num_users = mat["num_users"]
num_movies = mat["num_movies"]
num_features = mat["num_features"]

#  Reduce the data set size so that this runs faster
num_users = 4 
num_movies = 5 
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

#  Evaluate cost function
params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))
J, _ = ex8modules_cofi.cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)
           
print('Cost at loaded parameters: {:f} (this value should be about 22.22)\n'.format(J))

input('Program paused. Press enter to continue.')


## ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement 
#  the collaborative filtering gradient function.
#  
print('\nChecking Gradients (without regularization) ... \n')

#  Check gradients by running checkNNGradients
ex8utils.checkCostFunction()

## ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for 
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.
#  

#  Evaluate cost function
params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))
J, _ = ex8modules_cofi.cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)
           
print('Cost at loaded parameters (lambda_var = 1.5): {:f} (this value should be about 31.34)\n'.format(J))

## ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement 
#  regularization for the gradient. 
#

print('\nChecking Gradients (with regularization) ... \n')

#  Check gradients by running checkNNGradients
ex8utils.checkCostFunction(1.5)

input('Program paused. Press enter to continue.')


## ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
#

# Store all movies in movie list
n = 1682  # Total number of movies 
movieList = [None]*n
with open("homework\ex8\movie_ids.txt") as movie_ids_file:
    for i, line in enumerate(movie_ids_file.readlines()):
        movieName = line.split()[1:]
        movieList[i] = " ".join(movieName)

#  Initialize my ratings
my_ratings = np.zeros((1682, 1))

# NOTE THAT THE FOLLOWING SECTION AS WELL AS THE movie_ids.txt file
# USED HERE IS ADAPTED FOR PYTHON'S 0-INDEX (VS MATLAB/OCTAVE'S 1-INDEX)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 0, so to rate it "4", you can set
# my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
# my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[0] = 5
# my_ratings[6] = 3
my_ratings[11]= 5
# my_ratings[53] = 4
my_ratings[63]= 4
# my_ratings[65]= 3
my_ratings[68] = 3
my_ratings[97] = 4
# my_ratings[182] = 4
my_ratings[225] = 3
# my_ratings[354]= 5

print('\n\nNew user ratings:\n')
for i, rating in enumerate(my_ratings):
    if rating > 0: 
        print('Rated {:.0f} for {:s}'.format(rating[0], movieList[i]))


input('Program paused. Press enter to continue.')


## ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating 
#  dataset of 1682 movies and 943 users
#

print('\nTraining collaborative filtering...\n')

#  Load data
mat = scipy.io.loadmat('homework\ex8\ex8_movies.mat')
Y = mat["Y"]
R = mat["R"]

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = np.column_stack((my_ratings, Y))
R = np.column_stack(((my_ratings != 0).astype(int), R))


#  We normalize ratings here.
m, _ = Y.shape
Ymean = np.zeros((m, 1))
Ynorm = np.zeros(Y.shape)
for i in range(m):
    idx = R[i, :] == 1
    Ymean[i] = np.mean(Y[i, idx])
    Ynorm[i, idx] = Y[i, idx] - Ymean[i]

#  Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

# Set options
maxiter = 100
options = {'disp': True, 'maxiter':maxiter}
lambda_var=10

# Create "short hand" for the cost function to be minimized
def costFunc(initial_parameters):
    return ex8modules_cofi.cofiCostFunc(initial_parameters, Y, R, num_users, num_movies, num_features, lambda_var)

# Set Regularization
results = minimize(costFunc, x0=initial_parameters, options=options, method="L-BFGS-B", jac=True)
theta = results["x"]

# Unfold the returned theta back into U and W
X = np.reshape(theta[:num_movies*num_features], (num_movies, num_features), order='F')
Theta = np.reshape(theta[num_movies*num_features:], (num_users, num_features), order='F')

print('Recommender system learning complete.\n')


## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#

p = np.dot(X, Theta.T)
my_predictions = p[:,0] + Ymean.flatten()

# reverse sorting by index
ix = my_predictions.argsort()[::-1]

print('\n\nTop recommendations for you:\n')
for i in range(10):
    j = ix[i]
    print('Predicting rating {:.5f} for movie {}'.format(my_predictions[j], movieList[j]))

print('\n\nOriginal ratings provided:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated {:d} for {}'.format(int(my_ratings[i]), movieList[i]))
