import random
from nn_utils import *

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
        return self.layers[-1][0].a

    def get_layer_values(self, layer_idx):
        for neuron in self.layers[layer_idx]:
            yield neuron.a

    def propagate_backwards(self, desired_output):
        # TODO: brazko robim absolutny freestyle a vyzera ze to nefunguje(v test_nn neni uplne ze pekny vysledok) 
        # checknem to zajtra ale principialne som to robil podla vzorcov z tohoto
        # => https://brilliant.org/wiki/backpropagation/#:~:text=Backpropagation%2C%20short%20for%20%22backward%20propagation,to%20the%20neural%20network's%20weights.
        learning_rate = 0.2
        predicted_output = self.layers[-1][0].a
        for layer_idx in range(len(self.layers) - 1, 0, -1):
            for neuron_idx in range(len(self.layers[layer_idx])):
                # gradient = neuron_error * output(a) of the previous layer
                # new weight/bias value = old - learning_rate * gradient

                neuron = self.layers[layer_idx][neuron_idx]
                # calculate error for neuron in last layer
                if layer_idx == len(self.layers) - 1:
                    activation_dx = sigmoid_dx(neuron.z)
                    loss_dx = loss_function_dx(desired_output, predicted_output)
                    neuron.error = activation_dx * loss_dx

                    # calculate gradient for weights + adjust values
                    for weight_idx in range(len(neuron.weights)):
                        gradient = neuron.error * self.layers[layer_idx - 1][weight_idx].a
                        neuron.weights[weight_idx] -= learning_rate * gradient
                    # adjust bias
                    gradient = neuron.error
                    neuron.bias -= learning_rate * gradient
                    
                # calculate error for neuron in hidden layer
                else:
                    error_sum = 0
                    for neuron_next_idx in range(len(self.layers[layer_idx + 1])):
                        neuron_next = self.layers[layer_idx + 1][neuron_next_idx]
                        error_next = neuron_next.error
                        error_sum += error_next * neuron_next.weights[neuron_idx]
                    neuron.error = error_sum * sigmoid_dx(neuron.z)

                    # calculate gradient for weights + adjust values
                    for weight_idx in range(len(neuron.weights)):
                        gradient = neuron.error * self.layers[layer_idx - 1][weight_idx].a
                        neuron.weights[weight_idx] -= learning_rate * gradient
                    # adjust bias
                    gradient = neuron.error * neuron.bias
                    neuron.bias -= learning_rate * gradient
    
    def print_network(self):
        for i in range(len(self.layers)):
            print(f"Layer {i}:")
            for neuron in self.layers[i]:
                print(f"Bias: {neuron.bias}, value: {neuron.a}")

class Neuron:

    def __init__(self, prev_layer_wghts_count) -> None:
        self.weights = []
        self.error = 0
        for _ in range(prev_layer_wghts_count):
            self.weights.append(random.uniform(0, 1))
        self.bias = random.uniform(0, 1)

    def calc_value(self, previous_layer, is_output=False):
        z = 0
        for i, prev_value in enumerate(previous_layer):
            z += prev_value * self.weights[i]
        self.z = z + self.bias

        self.a = sigmoid(z)
        # if is_output:
        #     self.a = int(self.a > 0.5)
        # toto zaokruhlovanie budeme robit az pri uplnom "zverejnovani vysledku"... lebo zaujimaju nas aj male odchylky pocas ucenia
