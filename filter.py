import os
import utils
import re
import email as email_lib
from corpus import Corpus
from collections import Counter

class MyFiler:

    def __init__(self) -> None:
        params = {
            "relative_word_freq": 0,
            "name_number": 0,
            "mail_length": 0,
            "contains_html": 0,
            "in_blacklist": 0,
            "odd_sending_hours": 0,
            "hypertext": 0
        }
        pass
        
    def train(self, path):
        self.freq_spam = Counter()
        self.freq_ham = Counter()
        train_corpus = Corpus(path)
        truth = utils.read_classification_from_file(os.path.join(path, "!truth.txt"))
        for file_name, content in train_corpus.emails():
            if truth[file_name] == "OK":
                email = Email(content)
                self.freq_ham += email.word_frequencies_in_body()
            else:
                email = Email(content)
                self.freq_spam += email.word_frequencies_in_body()

        self.rel_freq = Counter()
        for key, value in self.freq_spam.most_common(100):
            self.rel_freq[key] = self.freq_spam[key] / (self.freq_spam[key] + self.freq_ham[key])
        print(self.rel_freq.most_common(100))
            

    def test(self, path):
        test_corpus = Corpus(path)
        predictions = {}
        for file_name, content in test_corpus.emails():
            predictions[file_name] = self.evaluate_email(content)
        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), predictions)

    def evaluate_email(self, content):
        email = Email(content)
        
        print(10*"-" + email.sender)
        print(email.body)
        print(email.subject, end="\n\n\n")
        
        return "OK"


class Email:
    def __init__(self, content) -> None:
        self.sender = None
        self.subject = None
        self.body = ""
        self.parse_email(content)

    def parse_email(self, content):
        email_object = email_lib.message_from_string(content)
        self.sender = email_object['from']
        self.subject = email_object['subject']
        
        # TODO: possibly a multipart mail => toto som dal takzvane ctrl-c ctrl-v
        if email_object.is_multipart():
            for part in email_object.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))

                # skip any text/plain (txt) attachments
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    self.body += part.get_payload()  # decode
                    break
        else:
            self.body = email_object.get_payload()
        # TODO: parse html
        self.body = re.sub(r"<[^>]*>", "", self.body)

    def word_frequencies_in_body(self):
        freq = Counter(self.body.lower().strip().split())
        return freq

