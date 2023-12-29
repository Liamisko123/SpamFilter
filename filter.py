import os
import utils
import nn_utils
import re
import random
import email as email_lib
from corpus import Corpus
from collections import Counter
from neuralnetwork import NN
from datetime import datetime
import quality
import string


def contains_html(text):
    pattern = r'<[^>]*>'
    return bool(re.search(pattern, text))

def contains_link(text):
    pattern = r'\b(?:https?|ftp):\/\/\S+'
    return bool(re.search(pattern, text))

NUMBER_OF_PARAMS = 7
NUMBER_OF_LAYERS = 2
NEURONS_IN_LAYER = 1000

class MyFiler:

    def __init__(self) -> None:
        self.network = NN(NUMBER_OF_PARAMS, NUMBER_OF_LAYERS, NEURONS_IN_LAYER)
        self.train_iters = 10000
        pass
        
    def train(self, path):
        self.freq_spam = Counter()
        self.freq_ham = Counter()
        truth = utils.read_classification_from_file(os.path.join(path, "!truth.txt"))
        truth_counter = Counter(truth.values())
        spam_count = truth_counter['SPAM']
        ham_count = truth_counter['OK']


        # Analyze parameters from dataset
        email_corpus = Corpus(path)
        for file_name, content in email_corpus.emails():
            if truth[file_name] == "OK":
                email = Email(content)
                self.freq_ham += email.word_frequencies_in_body()
            else:
                email = Email(content)
                self.freq_spam += email.word_frequencies_in_body()
        for key in self.freq_spam:
            self.freq_spam[key] /= spam_count
        
        for key in self.freq_ham:
            self.freq_ham[key] /= ham_count
        
        # print(self.freq_spam.most_common(100))
        # print()
        # print(self.freq_ham.most_common(100))
        # Train the neural network
        iter_count = 0
        all_params = []
        for file_name, content in email_corpus.emails():
            iter_count += 1
            email = Email(content)
            params = list(self.create_input(email).values())
            # train network on email
            target = (truth[file_name] == "SPAM")
            all_params.append((params, target))

        n_mails = len(all_params)
        i = 0
        for _ in range(self.train_iters):
            # print(all_params[i][0], int(all_params[i][1]))
            self.network.propagate_forward(all_params[i][0])
            self.network.propagate_backwards(int(all_params[i][1]))
            i += 1
            i = i % (n_mails-1)

    def test(self, path):
        print(40 * "-")
        test_corpus = Corpus(path)
        predictions = {}
        for file_name, content in test_corpus.emails():
            spamok = ["OK", "SPAM"]
            email = Email(content)
            params = list(self.create_input(email).values())
            predictions[file_name] = spamok[self.network.get_prediction(list(params))]
            # print(f"{self.network.get_output():.5f}", predictions[file_name])
        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), predictions)
        print(quality.compute_quality_for_corpus(path))

    def create_input(self, email):
        params = {
            "relative_word_freq": 0,
            "name_number": 0,
            "mail_length": 0.5,
            "contains_html": 0.5,
            "in_blacklist": 0.5,
            "odd_sending_hours": 0.5,
            "contains_link": 0.5
        }

        # common spam words
        mail_freq = email.word_frequencies_in_body()
        params["relative_word_freq"] = random.random()
        for key in mail_freq:
            params["relative_word_freq"] += mail_freq[key] * (self.freq_spam[key] - self.freq_ham[key])
        params["relative_word_freq"] = nn_utils.sigmoid(params["relative_word_freq"])
        # number in sender name
        num_count = 0
        for char in email.sender:
            if char in "1234567890":
                num_count += 1
        if num_count:
            params["name_number"] = 0.5 + (num_count / len(email.sender))
        mail_length = len(email.body)
        params["mail_length"] = mail_length / (mail_length + 1)

        params["contains_link"] = int(email.contains_link)

        params["contains_html"] = int(email.contains_html)
        # odd sending hours
        if email.time > 0 and email.time < 6:
            params["odd_sending_hours"] = nn_utils.sigmoid(1 - abs(email.time - 3)/3)
        return params

class Email:
    def __init__(self, content) -> None:
        self.sender = None
        self.subject = None
        self.contains_html = False
        self.contains_link = False
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
            self.contains_html = True
            self.body = re.sub(r"<[^>]*>", " ", self.body)
            self.body = re.sub(r"&nbsp", " ", self.body)

        if contains_link(self.body):
            self.contains_link = True

    def word_frequencies_in_body(self):
        alphabet = string.ascii_lowercase
        self.body = self.body.lower().strip()
        final_str = []
        for word in self.body.split():
            tmp_word = ""
            for char in word:
                if char in alphabet:
                    tmp_word += char
            if len(tmp_word):
                final_str.append(tmp_word)
        self.body = final_str
        freq = Counter(self.body)
        return freq

