## Machine Learning Online Class - Exercise 3

#  This file contains code that helps you get started on the logistic regression exercise.

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as inout

from ex3modules import *


## =========== Part 1: Loading and Visualizing Data =============
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.
data = inout.loadmat('homework\ex3\ex3data1.mat')
X = np.matrix(data['X'])
y = np.matrix(data['y'])

num_labels = 10          # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
m,n = X.shape
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100],:]

# closes previously opened figure. preventing a
# warning after opening too many figures
plt.close()

plt.ion()

# creates new figure 
plt.figure()

if type(sel)==np.matrix:
    sel = np.asarray(sel)

# turns 1D sel array into 2D
if sel.ndim == 1:
    sel = np.reshape(sel, (-1,sel.shape[0]))

# Set example_width automatically if not passed in
example_width = int(round(np.sqrt(sel.shape[1])))

# Gray Image
plt.set_cmap("gray")

# Compute rows, cols
m, n = sel.shape
example_height = int(n / example_width)

# Compute number of items to display
display_rows = int(np.floor(np.sqrt(m)))
display_cols = int(np.ceil(m / display_rows))

# Between images padding
pad = 1

# Setup blank display
display_array = -np.ones((pad + display_rows * (example_height + pad),  pad + display_cols * (example_width + pad)))

# Copy each example into a patch on the display array
curr_ex = 1
for j in range(1,display_rows+1):
    for i in range (1,display_cols+1):
        if curr_ex > m:
            break
    
        # Copy the patch
        
        # Get the max value of the patch to normalize all examples
        max_val = max(abs(sel[curr_ex-1, :]))
        rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
        cols = pad + (i - 1) * (example_width  + pad) + np.array(range(example_width ))

        display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = np.reshape(sel[curr_ex-1, :], (example_height, example_width), order="F") / max_val
        curr_ex += 1

    if curr_ex > m:
        break

# Display Image
h = plt.imshow(display_array, vmin=-1, vmax=1)

# Do not show axis
plt.axis('off')

print('\nDisplaying training data...')
plt.show()
input('Program paused. Press Enter to continue.')

## ============ Part 2: One-Vs-All Logistic Regression ============
print('\nTraining One-vs-All Logistic Regression...')
all_theta = trainOneVsAll(X, y, num_labels, 0.1)

pred = predictOneVsAll(all_theta, X)
print('Training Accuracy: {:.2f}%\n'.format(np.mean(pred == y) * 100))
