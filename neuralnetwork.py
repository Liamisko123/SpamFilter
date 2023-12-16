import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def sigmoid_dx(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

def relu_dx(x):
    return np.where(x > 0, 1, 0)

class NN:
    """Cievova siet"""

    def __init__(self) -> None:
        pass

    def evaluate(self):
        pass

    def train(self):
        pass
    
    def propagate_forward(self):
        pass

    def propagate_backwards(self):
        pass

    def loss_function(self):
        pass
    

