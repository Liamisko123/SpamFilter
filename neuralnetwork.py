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
        learning_rate = 0.2
        predicted_output = self.layers[-1][0].a
        for layer_idx in range(len(self.layers) - 1, 0, -1):
            for neuron_idx in range(len(self.layers[layer_idx])):
                neuron = self.layers[layer_idx][neuron_idx]
                # gradient = neuron_error * output(a) of the previous layer
                # new weight/bias value = old - learning_rate * gradient

                # calculate error for neuron in last layer
                if layer_idx == len(self.layers) - 1:
                    activation_dx = sigmoid_dx(neuron.a)
                    loss_dx = loss_function_dx(desired_output, predicted_output)
                    neuron.error = activation_dx * loss_dx
                    
                # calculate error for neuron in hidden layer
                else:
                    # TODO: Calculate error for non-last layer neuron
                    # na toto uz nemam nervy sry mate :D uvidim zajtra ci sa podari
                    # davam trz len freestyle
                    error = 0
                    for neuron_next_idx in range(len(self.layers[layer_idx + 1])):
                        neuron_next = self.layers[layer_idx + 1][neuron_next_idx]
                        error_next = neuron_next.error
                        a_dx = neuron_next.weights[neuron_idx] * sigmoid_dx(neuron.a)
                        error += error_next * a_dx
                    neuron.error = error
                    # end of freestyle pls check 
                    # => https://brilliant.org/wiki/backpropagation/#:~:text=Backpropagation%2C%20short%20for%20%22backward%20propagation,to%20the%20neural%20network's%20weights.
                    # presne v polovici tam pisu ako vypocitat error pre neuron v hidden layer

                # calculate gradient for weights + adjust values
                for weight_idx in range(len(neuron.weights)):
                    gradient = neuron.error * neuron.weights[weight_idx]
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
        if is_output:
            self.a = int(self.a > 0.5)
        # mozno sa to z aj a(namiesto len value) bude hodit... este to mozme zmenit ptm naspat ale niekde som cital ze to moze byt potrebne




