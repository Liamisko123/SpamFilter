from neuralnetwork import NN
import random

network = NN(2, 2, 5)
network.propagate_forward([random.uniform(0, 1) for _ in range(5)])
print(network.get_output())
# network.print_network()