# # Neural Networks for Planar Data Classification

# First, we import all the packages needed during this assignment:
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset

np.random.seed(1) # set a seed so that the results are consistent
plt.ion()

# You are given a randomly generated 2-class planar dataset that consists of...
#     - a numpy-array (matrix) X that contains your features (x1, x2)
#     - a numpy-array (vector) Y that contains your labels (red:0, blue:1).
# We are going to load this dataset visualize it.

X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
input('Program paused. Press Enter to continue.')

# For comparative experiment, we want to see how logistic regression performs on this problem.
# We will use the sklearn's built-in function to do this. Let's train this classifier here.
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

# We plot the decision boundary for logistic regression, and print out its accuracy.
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

input('Program paused. Press Enter to continue.')
# Accuracy is too low (i.e. below 50%) ? Don't worry. It is no surprise to see this poor result.
# This is because the dataset is not linearly separable, so the logistic regression doesn't perform well.

# To address this, we will build a Neural Network with the following steps:
#     1. Specify the neural network structure ( # of input units,  # of hidden units, etc). 
#     2. Define how to initialize the model's parameters
#     3. Define loop for the following tasks:
#         - forward propagation
#         - loss computation
#         - backward propagation
#         - gradient descent & parameter update

# **Exercise**: Implement the function `initialize_parameters()`.
# 
# You will initialize the weights matrices with random values.
# Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
# You will initialize the bias vectors as zeros.
# Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = 0.01 *np.random.random((n_h, n_x))
    b1 = np.zeros((n_h, 1))
    W2 = 0.01 * np.random.random((n_y, n_h))
    b2 = np.zeros((n_y, 1))
    ### END CODE HERE ###
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# **Exercise**: Implement `forward_propagation()`.
# - You can use the function `sigmoid()`. It is built-in (imported) in the notebook.
# - You can use the function `np.tanh()`. It is part of the numpy library.
# - The steps you have to implement are:
#     1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
#     2. Implement Forward Propagation. Compute $Z^{[1]}, A^{[1]}, Z^{[2]}$ and $A^{[2]}$ (the vector of all your predictions on all the examples in the training set).
# - Values needed in the backpropagation are stored in "`cache`". The `cache` will be given as an input to the backpropagation function.
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    ### END CODE HERE ###
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    ### START CODE HERE ### (≈ 4 lines of code)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    ### END CODE HERE ###
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)

# Note: we use the mean here just to make sure that your output matches ours. 
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))


# **Expected Output**:
# <table style="width:50%">
#   <tr>
#     <td> 0.262818640198 0.091999045227 -1.30766601287 0.212877681719 </td> 
#   </tr>
# </table>



# **Exercise**: Implement `compute_cost()` to compute the value of the cost $J$.
#
# BAD NEWS: You are required to do this in only 2 lines... WITHOUT using any loop.
# GOOD NEWS: I'll secretly give you a hint as follows:
# - Use `np.log()` on a matrix-type variable to compute element-wise logarithm values.
# - Use `np.multiply()` on a matrix-type variable to compute element-wise multiplication.
# - Use `np.dot()` on a matrix-type variable to compute matrix multiplication.
# - Use `np.sum()` on a matrix-type variable to compute element-wise addition.
# - Note that if you use `np.multiply` followed by `np.sum` the end result will be a type `float`, whereas if you use `np.dot`, the result will be a 2D numpy array. 
#   We can use `np.squeeze()` to remove redundant dimensions (in the case of single float, this will be reduced to a zero-dimension array). We can cast the array as a type `float` using `float()`.
def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    [Note that the parameters argument is not used in this function, 
    but the auto-grader currently expects this parameter.
    Future version of this notebook will fix both the notebook 
    and the auto-grader so that `parameters` is not needed.
    For now, please include `parameters` in the function signature,
    and also when invoking this function.]
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    ### START CODE HERE ### (≈ 2 lines of code)
    logprobs = (np.multiply(Y, np.log(A2)) + np.multiply((np.ones(Y.shape) - Y), np.log(np.ones(A2.shape) - A2))) / m
    cost = - np.sum(logprobs)
    ### END CODE HERE ###
    
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost




A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


# **Expected Output**:
# <table style="width:20%">
#   <tr>
#     <td>**cost**</td>
#     <td> 0.693058761... </td> 
#   </tr>
#   
# </table>

# Using the cache computed during forward propagation, you can now implement backward propagation.
# 
# **Question**: Implement the function `backward_propagation()`.
# 
# **Instructions**:
# Backpropagation is usually the hardest (most mathematical) part in deep learning. To help you, here again is the slide from the lecture on backpropagation. You'll want to use the six equations on the right of this slide, since you are building a vectorized implementation.  
# 
# <img src="images/grad_summary.png" style="width:600px;height:300px;">
# 
# <!--
# $\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } = \frac{1}{m} (a^{[2](i)} - y^{(i)})$
# 
# $\frac{\partial \mathcal{J} }{ \partial W_2 } = \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } a^{[1] (i) T} $
# 
# $\frac{\partial \mathcal{J} }{ \partial b_2 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)}}}$
# 
# $\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} } =  W_2^T \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } * ( 1 - a^{[1] (i) 2}) $
# 
# $\frac{\partial \mathcal{J} }{ \partial W_1 } = \frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} }  X^T $
# 
# $\frac{\partial \mathcal{J} _i }{ \partial b_1 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)}}}$
# 
# - Note that $*$ denotes elementwise multiplication.
# - The notation you will use is common in deep learning coding:
#     - dW1 = $\frac{\partial \mathcal{J} }{ \partial W_1 }$
#     - db1 = $\frac{\partial \mathcal{J} }{ \partial b_1 }$
#     - dW2 = $\frac{\partial \mathcal{J} }{ \partial W_2 }$
#     - db2 = $\frac{\partial \mathcal{J} }{ \partial b_2 }$
#     
# !-->
# 
# - Tips:
#     - To compute dZ1 you'll need to compute $g^{[1]'}(Z^{[1]})$. Since $g^{[1]}(.)$ is the tanh activation function, if $a = g^{[1]}(z)$ then $g^{[1]'}(z) = 1-a^2$. So you can compute 
#     $g^{[1]'}(Z^{[1]})$ using `(1 - np.power(A1, 2))`.



# GRADED FUNCTION: backward_propagation

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    ### START CODE HERE ### (≈ 2 lines of code)
    W1 = parameters['W1']
    W2 = parameters['W2']
    ### END CODE HERE ###
        
    # Retrieve also A1 and A2 from dictionary "cache".
    ### START CODE HERE ### (≈ 2 lines of code)
    A1 = cache['A1']
    A2 = cache['A2']
    ### END CODE HERE ###
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    ### END CODE HERE ###
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads




parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))


# **Expected output**:
# 
# 
# 
# <table style="width:80%">
#   <tr>
#     <td>**dW1**</td>
#     <td> [[ 0.00301023 -0.00747267]
#  [ 0.00257968 -0.00641288]
#  [-0.00156892  0.003893  ]
#  [-0.00652037  0.01618243]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**db1**</td>
#     <td>  [[ 0.00176201]
#  [ 0.00150995]
#  [-0.00091736]
#  [-0.00381422]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**dW2**</td>
#     <td> [[ 0.00078841  0.01765429 -0.00084166 -0.01022527]] </td> 
#   </tr>
#   
# 
#   <tr>
#     <td>**db2**</td>
#     <td> [[-0.16655712]] </td> 
#   </tr>
#   
# </table>  

# **Question**: Implement the update rule. Use gradient descent. You have to use (dW1, db1, dW2, db2) in order to update (W1, b1, W2, b2).
# 
# **General gradient descent rule**: $ \theta = \theta - \alpha \frac{\partial J }{ \partial \theta }$ where $\alpha$ is the learning rate and $\theta$ represents a parameter.
# 
# **Illustration**: The gradient descent algorithm with a good learning rate (converging) and a bad learning rate (diverging). Images courtesy of Adam Harley.
# 
# <img src="images/sgd.gif" style="width:400;height:400;"> <img src="images/sgd_bad.gif" style="width:400;height:400;">
# 
# 



# GRADED FUNCTION: update_parameters

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    ### END CODE HERE ###
    
    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### (≈ 4 lines of code)
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    ## END CODE HERE ###
    
    # Update rule for each parameter
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    ### END CODE HERE ###
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters




parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# **Expected Output**:
# 
# 
# <table style="width:80%">
#   <tr>
#     <td>**W1**</td>
#     <td> [[-0.00643025  0.01936718]
#  [-0.02410458  0.03978052]
#  [-0.01653973 -0.02096177]
#  [ 0.01046864 -0.05990141]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**b1**</td>
#     <td> [[ -1.02420756e-06]
#  [  1.27373948e-05]
#  [  8.32996807e-07]
#  [ -3.20136836e-06]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**W2**</td>
#     <td> [[-0.01041081 -0.04463285  0.01758031  0.04747113]] </td> 
#   </tr>
#   
# 
#   <tr>
#     <td>**b2**</td>
#     <td> [[ 0.00010457]] </td> 
#   </tr>
#   
# </table>  

# ### 4.4 - Integrate parts 4.1, 4.2 and 4.3 in nn_model() ####
# 
# **Question**: Build your neural network model in `nn_model()`.
# 
# **Instructions**: The neural network model has to use the previous functions in the right order.



# GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    # Initialize parameters
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (≈ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
        
        ### END CODE HERE ###
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters




X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# **Expected Output**:
# 
# <table style="width:90%">
# 
# <tr> 
#     <td> 
#         **cost after iteration 0**
#     </td>
#     <td> 
#         0.692739
#     </td>
# </tr>
# 
# <tr> 
#     <td> 
#         <center> $\vdots$ </center>
#     </td>
#     <td> 
#         <center> $\vdots$ </center>
#     </td>
# </tr>
# 
#   <tr>
#     <td>**W1**</td>
#     <td> [[-0.65848169  1.21866811]
#  [-0.76204273  1.39377573]
#  [ 0.5792005  -1.10397703]
#  [ 0.76773391 -1.41477129]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**b1**</td>
#     <td> [[ 0.287592  ]
#  [ 0.3511264 ]
#  [-0.2431246 ]
#  [-0.35772805]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**W2**</td>
#     <td> [[-2.45566237 -3.27042274  2.00784958  3.36773273]] </td> 
#   </tr>
#   
# 
#   <tr>
#     <td>**b2**</td>
#     <td> [[ 0.20459656]] </td> 
#   </tr>
#   
# </table>  

# ### 4.5 Predictions
# 
# **Question**: Use your model to predict by building predict().
# Use forward propagation to predict results.
# 
# **Reminder**: predictions = $y_{prediction} = \mathbb 1 \text{{activation > 0.5}} = \begin{cases}
#       1 & \text{if}\ activation > 0.5 \\
#       0 & \text{otherwise}
#     \end{cases}$  
#     
# As an example, if you would like to set the entries of a matrix X to 0 and 1 based on a threshold you would do: ```X_new = (X > threshold)```



# GRADED FUNCTION: predict

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X,parameters)
    predictions = (A2 > 0.5)
    ### END CODE HERE ###
    
    return predictions




parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))


# **Expected Output**: 
# 
# 
# <table style="width:40%">
#   <tr>
#     <td>**predictions mean**</td>
#     <td> 0.666666666667 </td> 
#   </tr>
#   
# </table>

# It is time to run the model and see how it performs on a planar dataset. Run the following code to test your model with a single hidden layer of $n_h$ hidden units.



# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
input('Program paused. Press Enter to continue.')

# **Expected Output**:
# 
# <table style="width:40%">
#   <tr>
#     <td>**Cost after iteration 9000**</td>
#     <td> 0.218607 </td> 
#   </tr>
#   
# </table>
# 



# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


# **Expected Output**: 
# 
# <table style="width:15%">
#   <tr>
#     <td>**Accuracy**</td>
#     <td> 90% </td> 
#   </tr>
# </table>

# Accuracy is really high compared to Logistic Regression. The model has learnt the leaf patterns of the flower! Neural networks are able to learn even highly non-linear decision boundaries, unlike logistic regression. 
# 
# Now, let's try out several hidden layer sizes.

# ### 4.6 - Tuning hidden layer size (optional/ungraded exercise) ###
# 
# Run the following code. It may take 1-2 minutes. You will observe different behaviors of the model for various hidden layer sizes.



# This may take about 2 minutes to run

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
input('Program paused. Press Enter to continue.')

# **Interpretation**:
# - The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data. 
# - The best hidden layer size seems to be around n_h = 5. Indeed, a value around here seems to  fits the data well without also incurring noticeable overfitting.
# - You will also learn later about regularization, which lets you use very large models (such as n_h = 50) without much overfitting. 
