from neuralnetwork import NN
import random

desired_output = False
network = NN(2, 2, 5)
data = [random.uniform(0, 1) for _ in range(5)]

network.propagate_forward(data)
print(network.get_output())
# network.print_network()

print("\n\n\n\n")
for i in range(1000):
    network.propagate_forward(data)
    network.propagate_backwards(desired_output)

print(network.get_output())
# network.print_network()