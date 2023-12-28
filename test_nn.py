from neuralnetwork import NN
import random
import time

desired_output = 1
desired_output2 = 1
desired_output3 = 0

network = NN(5, 2, 20)
#  relative.., name_number, contains_html, in_blacklist, hypertext
data1 = [0.8, 1, 1, 0, 0.65]
data2 = [0.2, 0, 1, 0, 0.95]
data3 = [0.4, 0, 0, 0, 0.20]

print(network.get_prediction(data1))
print(network.get_prediction(data2))
print(network.get_prediction(data3))
# network.print_network()

print("\n\n\n\n")


iters = 100
start = time.time()
for i in range(iters):

    # network.propagate_forward(data1)
    # network.propagate_backwards(desired_output)

    network.propagate_forward(data2)
    network.propagate_backwards(desired_output2)
    
    network.propagate_forward(data3)
    network.propagate_backwards(desired_output3)

end = time.time()
print(f"time: {end - start}")
print(f"after {iters} iterations")
network.propagate_forward(data1)
print(f"got {network.get_output():.4f}\twant: {desired_output}")
network.propagate_forward(data2)
print(f"got {network.get_output():.4f}\twant: {desired_output2}")
network.propagate_forward(data3)
print(f"got {network.get_output():.4f}\twant: {desired_output3}")

# network.print_network()