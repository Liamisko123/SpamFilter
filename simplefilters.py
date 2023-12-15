import os
from random import choice
from corpus import Corpus
import utils

class Filter:
    def __init__(self) -> None:
        pass
        
    def train(self, path):
        pass

    def test(self, path):
        pass

class NaiveFilter(Filter):
    def test(self, path):
        test_corpus = Corpus(path)
        predictions = {}
        for file_name, content in test_corpus.emails():
            ## some logic for <content> here ##
            predictions[file_name] = "OK"
        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), predictions)

        
class RandomFilter(Filter):
    def test(self, path):
        test_corpus = Corpus(path)
        predictions = {}
        for file_name, content in test_corpus.emails():
            predictions[file_name] = choice("SPAM", "OK")
        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), predictions)

class ParanoidFilter(Filter):
    def test(self, path):
        test_corpus = Corpus(path)
        predictions = {}
        for file_name, content in test_corpus.emails():
            predictions[file_name] = "SPAM"
        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), predictions)
