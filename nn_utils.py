import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def sigmoid_dx(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

def relu_dx(x):
    return np.where(x > 0, 1, 0)

def loss_function(output, target):
    return ((int(target) - int(output))**2)/2 # bad function for binary classification
    # (https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression) ? 
    # -> agreed ale zatial mozme nechat tento MSE a zmenime to ked to vobec nebude fungovat
    # nic by to vyrazne nemalo zmenit ked sa rozhodneme dat inu funkciu len proste tuto to zmenime a bude oukaj

def loss_function_dx(output, target):
    return int(target) - int(output)
