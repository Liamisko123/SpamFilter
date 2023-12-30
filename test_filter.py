import os
#from simplefilters import ParanoidFilter as filter
from filter import MyFiler as filter

new = filter()
# new.train(os.path.join(os.getcwd(), '1'), debug=False)
new.test(os.path.join(os.getcwd(), '1'), debug=False)
new.test(os.path.join(os.getcwd(), '2'))
