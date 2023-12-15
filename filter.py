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
        for file in test_corpus.emails():
            ## some logic for <content> here ##
            file_name, content = file
            predictions[file_name] = self.evaluate_email(content)
        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), predictions)

    def evaluate_email(self, content):
        email = Email(content)
        print(email.sender)


        return "OK"


class Email:
    def __init__(self, content) -> None:
        self.subject = None
        self.sender = None
        self.decompose_email(content)

    def decompose_email(self, content):
        #find stuff like sender and whatnot
        pass