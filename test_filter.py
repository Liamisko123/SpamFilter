import os
from filter import MyFilter as filter

new = filter()
new.train_network = True
new.train(os.path.join(os.getcwd(), '1'))
new.test(os.path.join(os.getcwd(), '1'))
new.test(os.path.join(os.getcwd(), '2'))
