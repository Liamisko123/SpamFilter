import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_dx(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

# def sigmoid_dx(x):
#     return x * (1 - x)


def loss_function(target, predicted):
    # return ((target - predicted)**2)/2 # bad function for binary classification
    # (https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression) ? 
    return binary_cross_entropy_loss(target, predicted)

def loss_function_dx(target, predicted):
    # return predicted - target
    return binary_cross_entropy_loss_derivative(target, predicted)

def binary_cross_entropy_loss(target, predicted):
    return - (target * np.log(predicted) + (1 - target) * np.log(1 - predicted))

def binary_cross_entropy_loss_derivative(target, predicted):
    epsilon = 1e-10
    return - (target / (predicted + epsilon)) + (1 - target) / (1 - (predicted + epsilon))

def relu_dx(x):
    return np.where(x > 0, 1, 0)

def relu(x):
    return np.maximum(0, x)
