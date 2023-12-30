import random

from nn_utils import *

class NN:
    """Cievova siet"""

    def __init__(self, in_params=1, number_of_layers=2, neurons_in_layer=2, learning_rate=0.2) -> None:
        self.layers = []
        self.learning_rate = learning_rate
        
        # Hidden layers
        for layer_idx in range(number_of_layers):
            tmp_layer = []
            if layer_idx ==  0: # layer connected to input layer
                for _ in range(neurons_in_layer):
                    tmp_layer.append(Neuron(in_params))
            else:
                for _ in range(neurons_in_layer):
                    tmp_layer.append(Neuron(neurons_in_layer))
            self.layers.append(tmp_layer)

        # Output
        self.layers.append([Neuron(neurons_in_layer)])
    
    def propagate_forward(self, input):
        for layer_idx in range(len(self.layers)):
            if layer_idx == 0:
                for neuron in self.layers[layer_idx]:
                    neuron.calc_value(input)
            else:
                prev_values = list(self.get_layer_values(layer_idx-1))
                for neuron in self.layers[layer_idx]:
                    neuron.calc_value(prev_values)
                prev_values = []

    def get_prediction(self, input):
        self.propagate_forward(input)
        return self.layers[-1][0].a > 0.5
    
    def get_output(self):
        return self.layers[-1][0].a

    def get_layer_values(self, layer_idx):
        for neuron in self.layers[layer_idx]:
            yield neuron.a

    def propagate_backwards(self, target):
        # => https://brilliant.org/wiki/backpropagation/#:~:text=Backpropagation%2C%20short%20for%20%22backward%20propagation,to%20the%20neural%20network's%20weights.
        prediction = self.layers[-1][0].a
        for layer_idx in range(len(self.layers) - 1, 0, -1):
            for neuron_idx in range(len(self.layers[layer_idx])):
                # gradient = neuron_error * output(a) of the previous layer
                # new weight/bias value = old - learning_rate * gradient

                neuron = self.layers[layer_idx][neuron_idx]
                # calculate error for neuron in last layer
                if layer_idx == len(self.layers) - 1:
                    activation_dx = sigmoid_dx(neuron.z)
                    loss_dx = loss_function_dx(target, prediction)
                    neuron.error = activation_dx * loss_dx
                    
                # calculate error for neuron in hidden layer
                else:
                    error_sum = sum(neuron_next.error * neuron_next.weights[neuron_idx] for neuron_next in self.layers[layer_idx + 1])
                    neuron.error = error_sum * sigmoid_dx(neuron.z)

                # calculate gradient for weights + adjust values
                for weight_idx in range(len(neuron.weights)):
                    gradient = neuron.error * self.layers[layer_idx - 1][weight_idx].a
                    neuron.weights[weight_idx] -= self.learning_rate * gradient
                # adjust bias
                gradient = neuron.error
                neuron.bias -= self.learning_rate * gradient
        # self.learning_rate /= 1.002
        # TODO: fine tuning learning rate
    
    def print_network(self):
        for i in range(len(self.layers)):
            print(f"Layer {i}:")
            for neuron in self.layers[i]:
                print(f"\tb: {neuron.bias}\tw: {[f'{w:.3f}' for w in neuron.weights]}\tz: {neuron.z}\ta: {neuron.a}")

class Neuron:

    def __init__(self, prev_layer_wghts_count) -> None:
        self.weights = []
        self.error = 0
        for _ in range(prev_layer_wghts_count):
            self.weights.append(random.uniform(-1, 1))
        self.bias = random.uniform(-1, 1)

    def calc_value(self, previous_layer):
        z = 0
        for i, prev_value in enumerate(previous_layer):
            z += prev_value * self.weights[i]
        self.z = z + self.bias

        self.a = sigmoid(z)
        # toto zaokruhlovanie budeme robit az pri uplnom "zverejnovani vysledku"... lebo zaujimaju nas aj male odchylky pocas ucenia
