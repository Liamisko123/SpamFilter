import os
#from simplefilters import ParanoidFilter as filter
from filter import MyFiler as filter
LOAD = 1
new = filter()
# new.test(os.path.join(os.getcwd(), '1'))
if LOAD:
    new.load_network()
new.train(os.path.join(os.getcwd(), '1'), True)
new.save_network()
new.test(os.path.join(os.getcwd(), '1'), True)
new.test(os.path.join(os.getcwd(), '2'))