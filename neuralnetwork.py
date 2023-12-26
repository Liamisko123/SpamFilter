import numpy as np
import random
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

    def __init__(self, number_of_layers=2, neurons_in_layer=2, in_params=1) -> None:
        self.layers = []

        # Input layer
        tmp_layer = []
        for _ in range(neurons_in_layer):
            tmp_layer.append(Neuron(in_params))
        self.layers.append(tmp_layer)
        
        # Hidden layers
        for _ in range(number_of_layers):
            tmp_layer = []
            for _ in range(neurons_in_layer):
                tmp_layer.append(Neuron(neurons_in_layer))
            self.layers.append(tmp_layer)

        # Output
        self.layers.append([Neuron(neurons_in_layer)])

    def evaluate(self):
        pass

    def train(self, input, target):
        self.propagate_forward(input)
        output = self.get_output()
        error = self.loss_function(output, target)
        self.propagate_backwards(error) # to be implemented
        print(f"Inputs: {input}")
        print(f"Target: {target}")
        print(f"Prediction: {output} (Error: {error})")
    
    def propagate_forward(self, input):
        layers_count = len(self.layers)

        for layer_idx in range(len(self.layers)):
            if layer_idx == 0:
                for neuron in self.layers[layer_idx]:
                    neuron.calc_value(input)
            else:
                is_output = (layer_idx == layers_count-1)
                prev_values = list(self.get_layer_values(layer_idx-1))
                for neuron in self.layers[layer_idx]:
                    neuron.calc_value(prev_values, is_output)
                prev_values = []

    def get_output(self):
        return self.layers[-1][0].value

    def get_layer_values(self, layer_idx):
        for neuron in self.layers[layer_idx]:
            yield neuron.value

    def propagate_backwards(self, error):
        pass

    def loss_function(self, output, target):
        return ((int(target) - int(output))**2)/2 # bad function for binary classification
    # (https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression) ?

    def print_network(self):
        for i in range(len(self.layers)):
            print(f"Layer {i}:")
            for neuron in self.layers[i]:
                print(f"Bias: {neuron.bias}, value: {neuron.value}")

class Neuron:

    def __init__(self, prev_layer_wghts_count) -> None:
        self.weights = []
        for _ in range(prev_layer_wghts_count):
            self.weights.append(random.uniform(0, 1))
        self.bias = random.uniform(0, 1)

    def calc_value(self, previous_layer, is_output=False):
        value = 0
        for i, prev_value in enumerate(previous_layer):
            value += prev_value * self.weights[i]
        value += self.bias

        self.value = sigmoid(value)
        if is_output:
            self.value = (value > 0.5)




