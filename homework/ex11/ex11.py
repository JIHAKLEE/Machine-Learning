# coding: utf-8

# # Building your Deep Neural Network: Step by Step
#
# First, we import all the packages needed during this assignment:
import numpy as np
import matplotlib.pyplot as plt

# The following is the set of activation functions necessary for constructing your deep neural network:
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache

def relu_backward(dA, cache): # backprop of relu
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    dZ[Z <= 0] = 0 # When z <= 0, you should set dz to 0 as well. 
    return dZ

def sigmoid_backward(dA, cache): # backprop of sigmoid
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


# To build your neural network, you will be implementing several "helper functions" to build a two-layer neural network and an L-layer neural network. Each small helper function you will implement will have detailed instructions that will walk you through the necessary steps. Here is an outline of this assignment, you will:
# 
# - Initialize the parameters for a two-layer network and for an $L$-layer neural network.
# - Implement the forward propagation module (shown in purple in the figure below).
#      - Complete the LINEAR part of a layer's forward propagation step (resulting in $Z^{[l]}$).
#      - We give you the ACTIVATION function (relu/sigmoid).
#      - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
#      - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). This gives you a new L_model_forward function.
# - Compute the loss.
# - Implement the backward propagation module (denoted in red in the figure below).
#     - Complete the LINEAR part of a layer's backward propagation step.
#     - We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward) 
#     - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
#     - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
# - Finally update the parameters.
# 
# **Note** that for every forward function, there is a corresponding backward function. That is why at every step of your forward module you will be storing some values in a cache. The cached values are useful for computing gradients. In the backpropagation module you will then use the cache to calculate the gradients. This assignment will show you exactly how to carry out each of these steps. 

# **Exercise**: Create and initialize the parameters of the 2-layer neural network.
# - The model's structure is: *LINEAR -> RELU -> LINEAR -> SIGMOID*. 
# - Use random initialization for the weight matrices. Use `np.random.randn(shape)*0.01` with the correct shape.
# - Use zero initialization for the biases. Use `np.zeros(shape)`.
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y,1))
    ### END CODE HERE ###
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    




parameters = initialize_parameters(3,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# **Expected output**:
#        
# <table style="width:80%">
#   <tr>
#     <td> **W1** </td>
#     <td> [[ 0.01624345 -0.00611756 -0.00528172]
#  [-0.01072969  0.00865408 -0.02301539]] </td> 
#   </tr>
# 
#   <tr>
#     <td> **b1**</td>
#     <td>[[ 0.]
#  [ 0.]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**W2**</td>
#     <td> [[ 0.01744812 -0.00761207]]</td>
#   </tr>
#   
#   <tr>
#     <td> **b2** </td>
#     <td> [[ 0.]] </td> 
#   </tr>
#   
# </table>

# **Exercise**: Implement initialization for an L-layer Neural Network. 
# - The model's structure is *[LINEAR -> RELU] $ \times$ (L-1) -> LINEAR -> SIGMOID*. I.e., it has $L-1$ layers using a ReLU activation function followed by an output layer with a sigmoid activation function.
# - Use random initialization for the weight matrices. Use `np.random.randn(shape) * 0.01`.
# - Use zeros initialization for the biases. Use `np.zeros(shape)`.
# - We will store $n^{[l]}$, the number of units in different layers, in a variable `layer_dims`. For example, the `layer_dims` for the "Planar Data classification model" from last week would have been [2,4,1]: There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit. This means `W1`'s shape was (4,2), `b1` was (4,1), `W2` was (1,4) and `b2` was (1,1). Now you will generalize this to $L$ layers! 
# - Here is the implementation for $L=1$ (one layer neural network). It should inspire you to implement the general case (L-layer neural network).
# ```python
#     if L == 1:
#         parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
#         parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))
# ```

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] =  np.zeros((layer_dims[l], 1))
        ### END CODE HERE ###
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# **Expected output**:
#        
# <table style="width:80%">
#   <tr>
#     <td> **W1** </td>
#     <td>[[ 0.01788628  0.0043651   0.00096497 -0.01863493 -0.00277388]
#  [-0.00354759 -0.00082741 -0.00627001 -0.00043818 -0.00477218]
#  [-0.01313865  0.00884622  0.00881318  0.01709573  0.00050034]
#  [-0.00404677 -0.0054536  -0.01546477  0.00982367 -0.01101068]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**b1** </td>
#     <td>[[ 0.]
#  [ 0.]
#  [ 0.]
#  [ 0.]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**W2** </td>
#     <td>[[-0.01185047 -0.0020565   0.01486148  0.00236716]
#  [-0.01023785 -0.00712993  0.00625245 -0.00160513]
#  [-0.00768836 -0.00230031  0.00745056  0.01976111]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**b2** </td>
#     <td>[[ 0.]
#  [ 0.]
#  [ 0.]]</td> 
#   </tr>
#   
# </table>

# Now that you have initialized your parameters, you will do the forward propagation module. You will start by implementing some basic functions that you will use later when implementing the model. You will complete three functions in this order:
# 
# - LINEAR
# - LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid. 
# - [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID (whole model)
# 
# **Exercise**: Build the linear part of forward propagation.
# Hint: Use `np.dot()` for matrix inner product multiplication. If your dimensions don't match, printing `W.shape` may help.
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W, A) + b
    ### END CODE HERE ###
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

# For more convenience, you are going to group two functions (Linear and Activation) into one function (LINEAR->ACTIVATION). Hence, you will implement a function that does the LINEAR forward step followed by an ACTIVATION forward step.
# 
# **Exercise**: Implement the forward propagation of the *LINEAR->ACTIVATION* layer.
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache  = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        ### END CODE HERE ###
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        ### END CODE HERE ###
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache



np.random.seed(2)
A_prev = np.random.randn(3,2)
W = np.random.randn(1,3)
b = np.random.randn(1,1)
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))


# **Expected output**:
#        
# <table style="width:35%">
#   <tr>
#     <td> **With sigmoid: A ** </td>
#     <td > [[ 0.96890023  0.11013289]]</td> 
#   </tr>
#   <tr>
#     <td> **With ReLU: A ** </td>
#     <td > [[ 3.43896131  0.        ]]</td> 
#   </tr>
# </table>
# 

# **Note**: In deep learning, the "[LINEAR->ACTIVATION]" computation is counted as a single layer in the neural network, not two layers. 

# For even more convenience when implementing the $L$-layer Neural Net, you will need a function that replicates the previous one (`linear_activation_forward` with RELU) $L-1$ times, then follows that with one `linear_activation_forward` with SIGMOID.
# **Exercise**: Implement the forward propagation of the above model.
# In the code below, the variable `AL` will denote $A^{[L]} = \sigma(Z^{[L]}) = \sigma(W^{[L]} A^{[L-1]} + b^{[L]})$. (This is sometimes also called `Yhat`, i.e., this is $\hat{Y}$.) 
# 
# **Tips**:
# - Use the functions you had previously written 
# - Use a for loop to replicate [LINEAR->RELU] (L-1) times
# - Don't forget to keep track of the caches in the "caches" list. To add a new value `c` to a `list`, you can use `list.append(c)`.


# GRADED FUNCTION: L_model_forward

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        ### START CODE HERE ### (≈ 2 lines of code)
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
        ### END CODE HERE ###
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ### START CODE HERE ### (≈ 2 lines of code)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    ### END CODE HERE ###
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


np.random.seed(6)
X = np.random.randn(5,4)
W1 = np.random.randn(4,5)
b1 = np.random.randn(4,1)
W2 = np.random.randn(3,4)
b2 = np.random.randn(3,1)
W3 = np.random.randn(1,3)
b3 = np.random.randn(1,1)

parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2,
                "W3": W3,
                "b3": b3}
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))


# <table style="width:50%">
#   <tr>
#     <td> **AL** </td>
#     <td > [[ 0.03921668  0.70498921  0.19734387  0.04728177]]</td> 
#   </tr>
#   <tr>
#     <td> **Length of caches list ** </td>
#     <td > 3 </td> 
#   </tr>
# </table>

# **Exercise**: Compute the cross-entropy cost $J$
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    logprobs = np.multiply(np.log(AL),Y) +  np.multiply(np.log(1-AL), (1-Y))
    cost = -1/m*np.sum(logprobs)
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost



Y = np.asarray([[1, 1, 1]])
AL = np.array([[.8,.9,0.4]])
print("cost = " + str(compute_cost(AL, Y)))


# **Expected Output**:
# 
# <table>
# 
#     <tr>
#     <td>**cost** </td>
#     <td> 0.2797765635793422</td> 
#     </tr>
# </table>

# Back propagation is used to calculate the gradient of the loss function with respect to the parameters. 
# Similar to forward propagation, you are going to build the backward propagation in three steps:
# - LINEAR backward
# - LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation
# - [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID backward (whole model)


# **Exercise**: implement linear_backward().

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = 1./m*np.dot(dZ, A_prev.T)
    db = 1./m*np.sum(dZ, axis = 1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    ### END CODE HERE ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db




# Next, you will create a function that merges the two helper functions: **`linear_backward`** and the backward step for the activation **`linear_activation_backward`**. 
# 
# To help you implement `linear_activation_backward`, we provided two backward functions:
# - **`sigmoid_backward`**: Implements the backward propagation for SIGMOID unit.
# - **`relu_backward`**: Implements the backward propagation for RELU unit.
# **Exercise**: Implement the backpropagation for the *LINEAR->ACTIVATION* layer.
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
        
    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
    
    return dA_prev, dW, db




np.random.seed(2)
dAL = np.random.randn(1,2)
A = np.random.randn(3,2)
W = np.random.randn(1,3)
b = np.random.randn(1,1)
Z = np.random.randn(1,2)
linear_cache = (A, W, b)
activation_cache = Z
linear_activation_cache = (linear_cache, activation_cache)

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


# **Expected output with sigmoid:**
# 
# <table style="width:100%">
#   <tr>
#     <td > dA_prev </td> 
#            <td >[[ 0.11017994  0.01105339]
#  [ 0.09466817  0.00949723]
#  [-0.05743092 -0.00576154]] </td> 
# 
#   </tr> 
#   
#     <tr>
#     <td > dW </td> 
#            <td > [[ 0.10266786  0.09778551 -0.01968084]] </td> 
#   </tr> 
#   
#     <tr>
#     <td > db </td> 
#            <td > [[-0.05729622]] </td> 
#   </tr> 
# </table>
# 
# 

# **Expected output with relu:**
# 
# <table style="width:100%">
#   <tr>
#     <td > dA_prev </td> 
#            <td > [[ 0.44090989  0.        ]
#  [ 0.37883606  0.        ]
#  [-0.2298228   0.        ]] </td> 
# 
#   </tr> 
#   
#     <tr>
#     <td > dW </td> 
#            <td > [[ 0.44513824  0.37371418 -0.10478989]] </td> 
#   </tr> 
#   
#     <tr>
#     <td > db </td> 
#            <td > [[-0.20837892]] </td> 
#   </tr> 
# </table>
# 
# 

# **Exercise**: Implement backpropagation for the *[LINEAR->RELU] $\times$ (L-1) -> LINEAR -> SIGMOID* model.
# To backpropagate through this network, your code needs to compute `dAL` $= \frac{\partial \mathcal{L}}{\partial A^{[L]}}$.
# You can then use this post-activation gradient `dAL` to keep going backward.
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    ### END CODE HERE ###
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    ### END CODE HERE ###
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)],  current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads



np.random.seed(3)
AL = np.random.randn(1, 2)
Y_assess = np.array([[1, 0]])

A1 = np.random.randn(4,2)
W1 = np.random.randn(3,4)
b1 = np.random.randn(3,1)
Z1 = np.random.randn(3,2)
linear_cache_activation_1 = ((A1, W1, b1), Z1)

A2 = np.random.randn(3,2)
W2 = np.random.randn(1,3)
b2 = np.random.randn(1,1)
Z2 = np.random.randn(1,2)
linear_cache_activation_2 = ((A2, W2, b2), Z2)

caches = (linear_cache_activation_1, linear_cache_activation_2)

grads = L_model_backward(AL, Y_assess, caches)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dA1 = "+ str(grads["dA1"]))


# **Expected Output**
# 
# <table style="width:60%">
#   
#   <tr>
#     <td > dW1 </td> 
#            <td > [[ 0.41010002  0.07807203  0.13798444  0.10502167]
#  [ 0.          0.          0.          0.        ]
#  [ 0.05283652  0.01005865  0.01777766  0.0135308 ]] </td> 
#   </tr> 
#   
#     <tr>
#     <td > db1 </td> 
#            <td > [[-0.22007063]
#  [ 0.        ]
#  [-0.02835349]] </td> 
#   </tr> 
#   
#   <tr>
#   <td > dA1 </td> 
#            <td > [[ 0.12913162 -0.44014127]
#  [-0.14175655  0.48317296]
#  [ 0.01663708 -0.05670698]] </td> 
# 
#   </tr> 
# </table>
# 
# 

# **Exercise**: Implement `update_parameters()` to update your parameters using gradient descent.
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]
    ### END CODE HERE ###
    return parameters




np.random.seed(2)
W1 = np.random.randn(3,4)
b1 = np.random.randn(3,1)
W2 = np.random.randn(1,3)
b2 = np.random.randn(1,1)
parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2}
np.random.seed(3)
dW1 = np.random.randn(3,4)
db1 = np.random.randn(3,1)
dW2 = np.random.randn(1,3)
db2 = np.random.randn(1,1)
grads = {"dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2}
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))


# **Expected Output**:
# 
# <table style="width:100%"> 
#     <tr>
#     <td > W1 </td> 
#            <td > [[-0.59562069 -0.09991781 -2.14584584  1.82662008]
#  [-1.76569676 -0.80627147  0.51115557 -1.18258802]
#  [-1.0535704  -0.86128581  0.68284052  2.20374577]] </td> 
#   </tr> 
#   
#     <tr>
#     <td > b1 </td> 
#            <td > [[-0.04659241]
#  [-1.28888275]
#  [ 0.53405496]] </td> 
#   </tr> 
#   <tr>
#     <td > W2 </td> 
#            <td > [[-0.55569196  0.0354055   1.32964895]]</td> 
#   </tr> 
#   
#     <tr>
#     <td > b2 </td> 
#            <td > [[-0.84610769]] </td> 
#   </tr> 
# </table>
#