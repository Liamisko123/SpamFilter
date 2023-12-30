import os
import utils
import nn_utils
import re
import random
import email as email_lib
import quality
import string
import pickle
from corpus import Corpus
from collections import Counter
from neuralnetwork import NN


def contains_html(text):
    pattern = r'<[^>]*>'
    return bool(re.search(pattern, text))

def contains_link(text):
    pattern = r'\b(?:https?|ftp):\/\/\S+'
    return bool(re.search(pattern, text))

NUMBER_OF_PARAMS = 7
NUMBER_OF_LAYERS = 2
NEURONS_IN_LAYER = 10

class MyFiler:

    def __init__(self) -> None:
        self.network = NN(NUMBER_OF_PARAMS, NUMBER_OF_LAYERS, NEURONS_IN_LAYER)
        self.train_iters = 100
        self.trained = False
        

    def save_network(self):
        with open("neural_network.pickle", "wb") as file:
            pickle.dump(self.network, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    def load_network(self):
        with open("neural_network.pickle", "rb") as file:
            self.network = pickle.load(file)
        
    def train(self, path, debug=False):
        truth = utils.read_classification_from_file(os.path.join(path, "!truth.txt"))
        train_corpus = Corpus(path)

        self.get_dataset_word_freqs(truth, train_corpus)

        # Get input params and target outputs for each email 
        all_params = []
        for file_name, content in train_corpus.emails():
            email = Email(content)
            in_params = list(self.create_input(email).values())
            target = (truth[file_name] == "SPAM")
            all_params.append((in_params, target))

        # Train the network
        print(f"Training on each email in {path} {self.train_iters} times...")
        n_mails = len(all_params)
        for i in range(self.train_iters):
            if debug:
                print("Training iteration", i+1)
                print("Learning rate:", self.network.learning_rate)
            for m in range(n_mails):
                self.network.propagate_forward(all_params[m][0])
                self.network.propagate_backwards(int(all_params[m][1]))
            self.network.learning_rate *= 0.997
        
        self.trained = True

    def test(self, path, debug=False):
        if not self.trained:
            print("The filter has not been trained!")
            return
        
        print(40 * "-")
        test_corpus = Corpus(path)
        predictions = {}
        for file_name, content in test_corpus.emails():
            spamok = ["OK", "SPAM"]
            email = Email(content)
            params = list(self.create_input(email).values())
            predictions[file_name] = spamok[self.network.get_prediction(params)]
            if debug:
                for i in range(len(params)):
                    params[i] = float(round(params[i], 2))
                print(f"{file_name:.7s}…: {params}    \t{self.network.get_output():.5f}")

        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), predictions)
        print(f"Classification quality for {path}:")
        print(quality.compute_quality_for_corpus(path))

    def create_input(self, email):
        params = {
            "relative_word_freq": 0,
            "name_number": 0,
            "mail_length": 0.5,
            "contains_html": 0.5,
            "in_blacklist": 0.5,
            "odd_sending_hours": 0,
            "contains_link": 0.5
        }

        # common spam words
        mail_freq = email.word_frequencies_in_body()
        params["relative_word_freq"] = random.random()
        words_c = 1
        for key in mail_freq:
            words_c += mail_freq[key]
            params["relative_word_freq"] += self.rel_freq[key] * mail_freq[key]
        params["relative_word_freq"] = nn_utils.sigmoid(3 * params["relative_word_freq"] / words_c)


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
        if email.time >= 0 and email.time <= 6:
            params["odd_sending_hours"] = nn_utils.sigmoid(1 - abs(email.time - 3)/4)
        return params
    
    def get_dataset_word_freqs(self, truth, corpus):
        truth_counter = Counter(truth.values())
        spam_count = truth_counter['SPAM']
        ham_count = truth_counter['OK']

        freq_spam = Counter()
        freq_ham = Counter()
        for file_name, content in corpus.emails():
            if truth[file_name] == "OK":
                email = Email(content)
                freq_ham += email.word_frequencies_in_body()
            else:
                email = Email(content)
                freq_spam += email.word_frequencies_in_body()
        for key in freq_spam:
            freq_spam[key] /= spam_count        
        for key in freq_ham:
            freq_ham[key] /= ham_count
        
        self.rel_freq = Counter()
        for word in freq_spam.most_common(1000):
            i = word[0]
            # print(i, end='; ')
            self.rel_freq[i] = (freq_spam[i]-freq_ham[i]) / (freq_spam[i]+freq_ham[i])
        
        # print(self.rel_freq.most_common(100))
        

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

