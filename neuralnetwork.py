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

    def __init__(self, number_of_layers, neurons_in_layer, in_params) -> None:
        self.layers = []

        tmp_layer = []
        for _ in range(neurons_in_layer):
            tmp_layer.append(Neuron(in_params))
        self.layers.append(tmp_layer)
        

        for _ in range(number_of_layers):
            tmp_layer = []
            for _ in range(neurons_in_layer):
                tmp_layer.append(Neuron(neurons_in_layer))
            self.layers.append(tmp_layer)

        self.layers.append([Neuron(neurons_in_layer)])

    def evaluate(self):
        pass

    def train(self):
        pass
    
    def propagate_forward(self, input):
        for neuron in self.layers[0]:
            neuron.calc_value(input)
        
        for layer_idx in range(1, len(self.layers)):
            prev_values = list(self.get_layer_values(layer_idx-1))
            for neuron in self.layers[layer_idx]:
                neuron.calc_value(prev_values)
            prev_values = []

    def get_output(self):
        return self.layers[-1][0].value

    def get_layer_values(self, layer_idx):
        for neuron in self.layers[layer_idx]:
            yield neuron.value

    def propagate_backwards(self):
        pass

    def loss_function(self):
        pass
    
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

    def calc_value(self, previous_layer):
        value = 0
        for i, prev_value in enumerate(previous_layer):
            value += prev_value * self.weights[i]
        value += self.bias

        self.value = sigmoid(value)



