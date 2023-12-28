import os
import utils
import re
import email as email_lib
from corpus import Corpus
from collections import Counter
from neuralnetwork import NN
from datetime import datetime
import quality


def contains_html(input_string):
    pattern = re.compile(r'<[^>]*>')
    return bool(pattern.search(input_string))

NUMBER_OF_PARAMS = 7
class MyFiler:

    def __init__(self) -> None:
        self.network = NN(NUMBER_OF_PARAMS)
        self.train_iters = 10000
        pass
        
    def train(self, path):
        train_predictions = {}

        self.freq_spam = Counter()
        self.freq_ham = Counter()
        truth = utils.read_classification_from_file(os.path.join(path, "!truth.txt"))

        # Analyze parameters from dataset
        email_corpus = Corpus(path)
        for file_name, content in email_corpus.emails():
            if truth[file_name] == "OK":
                email = Email(content)
                self.freq_ham += email.word_frequencies_in_body()
            else:
                email = Email(content)
                self.freq_spam += email.word_frequencies_in_body()

        self.rel_freq = Counter()
        for key, value in self.freq_spam.most_common(100):
            self.rel_freq[key] = self.freq_spam[key] / (self.freq_spam[key] + self.freq_ham[key])
        # print(self.rel_freq.most_common(100))
        

        # Train the neural network
        iter_count = 0
        for file_name, content in email_corpus.emails():
            iter_count += 1
            email = Email(content)
            params = list(self.create_input(email).values())
            # train network on email
            target = (truth[file_name] == "SPAM")
            for _ in range(self.train_iters):
                self.network.propagate_forward(params)
                self.network.propagate_backwards(target)
            spamok = ["OK", "SPAM"]
            train_predictions[file_name] = spamok[self.network.get_prediction(params)]
            print(train_predictions[file_name])

           
        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), train_predictions)
        print(quality.compute_quality_for_corpus(path))
        
            
    

    def test(self, path):
        test_corpus = Corpus(path)
        predictions = {}
        for file_name, content in test_corpus.emails():
            spamok = ["OK", "SPAM"]
            email = Email(content)
            params = list(self.create_input(email).values())
            predictions[file_name] = spamok[self.network.get_prediction(list(params))]
        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), predictions)
        print(quality.compute_quality_for_corpus(path))


    def create_input(self, email):
        params = {
            "relative_word_freq": 0,
            "name_number": 0,
            "mail_length": 0,
            "contains_html": 0,
            "in_blacklist": 0,
            "odd_sending_hours": 0,
            "hypertext": 0
        }

        # common spam words
        mail_freq = email.word_frequencies_in_body()
        count = 1
        params["relative_word_freq"] = 0
        for key in mail_freq:
            if key in self.rel_freq:
                count += mail_freq[key]
            params["relative_word_freq"] += self.rel_freq[key] * mail_freq[key]
        params["relative_word_freq"] /= count

        # number in sender name
        for char in email.sender:
            if char in "1234567890":
                params["name_number"] = 1
                break

        # odd sending hours
        if email.time > 0 and email.time < 6:
            params["odd_sending_hours"] = 1 - abs(email.time - 3)/3
        return params

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
        time_xyz = email_object['date'].split(":")
        h = time_xyz[0][-2:]
        self.time = int(h)
        # print(self.time)
        # self.time = ...
        
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
        if contains_html(self.body):
            self.body = re.sub(r"<[^>]*>", "", self.body)

    def word_frequencies_in_body(self):
        freq = Counter(self.body.lower().strip().split())
        return freq

