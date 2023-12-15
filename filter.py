import os
import utils
from corpus import Corpus


class MyFiler:

    def __init__(self) -> None:
        pass
        
    def train(self, path):
        pass

    def test(self, path):
        test_corpus = Corpus(path)
        predictions = {}
        for file_name, content in test_corpus.emails():
            ## some logic for <content> here ##
            predictions[file_name] = "OK"
        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), predictions)