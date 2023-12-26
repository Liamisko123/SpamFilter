from neuralnetwork import NN
import random

desired_output = 0
desired_output2 = 0
desired_output3 = 1

network = NN(2, 2, 5)
data = [random.uniform(0, 1) for _ in range(5)]
data2 = [random.uniform(0, 1) for _ in range(5)]
data3 = [random.uniform(0, 1) for _ in range(5)]

network.propagate_forward(data)
print(network.get_output())
network.propagate_forward(data2)
print(network.get_output())
network.propagate_forward(data3)
print(network.get_output())
# network.print_network()

print("\n\n\n\n")

iters = 100000
for i in range(iters):
    network.propagate_forward(data)
    if i == iters - 1:
        print(f"after {iters} iterations")
        print(f"got {network.get_output()}\trounded to {int(network.get_output() > 0.5)}\twant: {desired_output}")
    network.propagate_backwards(desired_output)

    network.propagate_forward(data2)
    if i == iters - 1:
        print(f"got {network.get_output()}\trounded to {int(network.get_output() > 0.5)}\twant: {desired_output2}")
    network.propagate_backwards(desired_output2)

    network.propagate_forward(data3)
    if i == iters - 1:
        print(f"got {network.get_output()}\trounded to {int(network.get_output() > 0.5)}\twant: {desired_output3}")
    network.propagate_backwards(desired_output3)
# TODO: vsimni si ze vysledky sa nekonecne priblizuju k 1 => nicht gut

# network.print_network()