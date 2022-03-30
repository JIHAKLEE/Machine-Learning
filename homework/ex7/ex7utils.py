import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np

import ex7modules_kmeans

def drawLine(p1, p2, **kwargs):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    # This function plots the data
    # points with colors assigned to each centroid. With the previous
    # centroids, it also plots a line between the previous locations and
    # current locations of the centroids.
    #

    # Plot the examples
    palette = clrs.hsv_to_rgb( np.column_stack([ np.linspace(0, 1, K+1), np.ones( ((K+1), 2) ) ]) ) # Create palette
    colors = np.array([palette[int(i)] for i in idx]) # Choose colors
    plt.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors=colors) # Plot the data

    # Plot the centroids as black x's
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=400, c='k', linewidth=1)

    # Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        drawLine(centroids[j, :], previous[j, :], c='b')

    # Title
    plt.title('Iteration number {:d}'.format(i+1))


def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    # This function runs the K-Means algorithm on data matrix X, where each 
    # row of X is a single example. It uses initial_centroids used as the
    # initial centroids. max_iters specifies the total number of interactions 
    # of K-Means to execute. plot_progress is a true/false flag that 
    # indicates if the function should also plot its progress as the 
    # learning happens. This is set to false by default. runkMeans returns 
    # centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
    # vector of centroid assignments (i.e. each entry in range [1..K])
    #

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))

    # if plotting, set up the space for interactive graphs
    # http://stackoverflow.com/a/4098938/583834
    # http://matplotlib.org/faq/usage_faq.html#what-is-interactive-mode
    if plot_progress:
        plt.close()
        plt.ion()

    # Run K-Means
    for i in range(max_iters):
        
        # Output progress
        print('K-Means iteration {:d}/{:d}...'.format(i+1, max_iters))
        
        # For each example in X, assign it to the closest centroid
        idx = ex7modules_kmeans.findClosestCentroids(X, centroids)
        
        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            input('Press enter to continue.')
        
        # Given the memberships, compute new centroids
        centroids = ex7modules_kmeans.computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    print('\n')

    return centroids, idx


def featureNormalize(X):
    #FEATURENORMALIZE Normalizes the features in X 
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.

    mu = np.mean(X, axis=0)
    X_norm = X - mu

    # note that a difference here with the matlab/octave way of handling
    # stddev produces different results further down the pipeline
    # see:
    #   http://stackoverflow.com/q/27600207/583834
    #   https://www.gnu.org/software/octave/doc/v4.0.3/Descriptive-Statistics.html#XREFstd
    # python's np.std() outputs:
    #   [ 1.16126017  1.01312201]
    # octave's std() outputs:
    #   [1.17304991480488,  1.02340777859473]
    # code below uses python np.std(..., ddof=1) following
    #   http://stackoverflow.com/a/27600240/583834
    sigma = np.std(X_norm, axis=0, ddof=1)

    X_norm = X_norm/sigma

    return X_norm, mu, sigma

    # ============================================================

def displayData(X, example_width=None):
    # This function displays 2D data stored in X in a nice grid.
    # It returns the figure handle h and the displayed array if requested.

    # using plt.ion() instead of the commented section below
    # # closes previously opened figure. preventing a
    # # warning after opening too many figures
    # plt.close()

    # # creates new figure 
    # plt.figure()

    # turns 1D X array into 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1,X.shape[0]))

    # Set example_width automatically if not passed in
    if not example_width or not 'example_width' in locals():
        example_width = int(round(np.sqrt(X.shape[1])))

    # Gray Image
    plt.set_cmap("gray")

    # Compute rows, cols
    m, n = X.shape
    example_height = int(round(n / example_width))

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
            max_val = max(abs(X[curr_ex-1, :]))
            rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
            cols = pad + (i - 1) * (example_width  + pad) + np.array(range(example_width ))

            # Basic (vs. advanced) indexing/slicing is necessary so that we look can assign
            #   values directly to display_array and not to a copy of its subarray.
            display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = np.reshape(X[curr_ex-1, :], (example_height, example_width), order="F") / max_val
            curr_ex += 1
    
        if curr_ex > m:
            break

    # Display Image
    h = plt.imshow(display_array, vmin=-1, vmax=1)

    # Do not show axis
    plt.axis('off')

    plt.show(block=False)

    return h, display_array
