import quality
import os
from sys import argv

print(quality.compute_quality_for_corpus(os.path.join(os.getcwd(), argv[1])))
