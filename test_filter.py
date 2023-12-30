import os
#from simplefilters import ParanoidFilter as filter
from filter import MyFiler as filter

new = filter()
new.load_network() # Delete save file to restart network
new.train(os.path.join(os.getcwd(), '1'), debug=False)
new.save_network()
new.test(os.path.join(os.getcwd(), '1'), debug=False)
new.test(os.path.join(os.getcwd(), '2'))
