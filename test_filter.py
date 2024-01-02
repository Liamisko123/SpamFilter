import os
from filter import MyFilter as filter

new = filter()
new.train_network = True
new.train(os.path.join(os.getcwd(), '1'))
new.train(os.path.join(os.getcwd(), '2'))
new.test(os.path.join(os.getcwd(), '1'), debug=True)
new.test(os.path.join(os.getcwd(), '2'), debug=True)
