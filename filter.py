import os
import utils
import re
import email as email_lib
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
            predictions[file_name] = self.evaluate_email(content)
        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), predictions)

    def evaluate_email(self, content):
        email = Email(content)
        print(10*"-" + email.sender)
        print(email.subject, end="\n\n\n")
        # print(email.body)

        return "OK"


class Email:
    def __init__(self, content) -> None:
        self.sender = None
        self.subject = None
        self.body = None
        self.parse_email(content)

    def parse_email(self, content):
        email_object = email_lib.message_from_string(content)
        self.sender = email_object['from']
        self.subject = email_object['subject']
        
        # TODO: possibly a multipart mail
        # if email_object.is_multipart():....
        self.body = email_object.get_payload()
        # TODO: parse html
        # self.body = re.sub("<.*>", "", self.body)

