import os
#from simplefilters import ParanoidFilter as filter
from filter import MyFiler as filter

new = filter()
new.test(os.path.join(os.getcwd(), '1'))
