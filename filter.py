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

NUMBER_OF_PARAMS = 7
NUMBER_OF_LAYERS = 2
NEURONS_IN_LAYER = 10
LEARNING_RATE = 0.2
# LEARNING_SLOWDOWN = 0.997

class MyFilter: 

    def __init__(self) -> None:
        self.network = NN(NUMBER_OF_PARAMS, NUMBER_OF_LAYERS, NEURONS_IN_LAYER, LEARNING_RATE)
        self.train_iters = 50
        self.train_network = False
        self.rel_freq = None
        self.load_network()
        self.load_filter_data()
    
    def save_network(self):
        with open("neural_network.pickle", "wb") as file:
            pickle.dump(self.network, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    def load_network(self): 
        try:
            with open("neural_network.pickle", "rb") as file:
                self.network = pickle.load(file)
        except FileNotFoundError:
            print("Network data file not found.")

    def save_filter_data(self):
        with open("filter_data.pickle", "wb") as file:
            pickle.dump(self.rel_freq, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_filter_data(self): 
        try:
            with open("filter_data.pickle", "rb") as file:
                self.rel_freq = pickle.load(file) 
        except FileNotFoundError:
            print("Filter data file not found.")

    def train(self, path, debug=False):
        self.loaded_network = True
        truth = utils.read_classification_from_file(os.path.join(path, "!truth.txt"))
        train_corpus = Corpus(path)
        

        if not self.rel_freq:
            self.get_dataset_word_freqs(truth, train_corpus)
        
        if self.train_network:   
            # Get input params and target outputs for each email 
            all_params = []
            for file_name, content in train_corpus.emails():
                email = Email(content)
                in_params = list(self.create_input(email).values())
                target = (truth[file_name] == "SPAM")
                all_params.append((in_params, target))
                
            # Train the network
            if debug:
                print(f"Training on each email in {path} {self.train_iters} times...")
            n_mails = len(all_params)
            for i in range(self.train_iters):
                if debug == "all":
                    print("Training iteration", i+1)
                for m in range(n_mails):
                    self.network.propagate_forward(all_params[m][0])
                    self.network.propagate_backwards(int(all_params[m][1]))
                # self.network.learning_rate *= LEARNING_SLOWDOWN
            if debug:
                print("Learning rate:", self.network.learning_rate)
            self.save_network()

        self.save_filter_data()


    def test(self, path, debug=False):
        if debug:
            print(40 * "-")
        test_corpus = Corpus(path)
        predictions = {}
        for file_name, content in test_corpus.emails():
            spamok = ["OK", "SPAM"]
            email = Email(content)
            params = list(self.create_input(email).values())
            predictions[file_name] = spamok[self.network.get_prediction(params)]
            if debug == "all":
                for i in range(len(params)):
                    params[i] = float(round(params[i], 2))
                print(f"{file_name:.7s}…: {params}    \t{self.network.get_output():.5f}")

        utils.write_classification_to_file(os.path.join(path, "!prediction.txt"), predictions)
        if debug:
            print(f"Classification quality for {path}:")
            q = quality.compute_quality_for_corpus(path)
            print(q)
            if q == 0.23154193872425916:
                print("Warning: all entries were flagged as SPAM.")
            elif q == 0.249185667752443:
                print("Warning: all entries were flagged as OK.")



    def create_input(self, email):
        params = {
            "relative_word_freq": 0,
            "name_number": 0,
            "mail_length": 0.5,
            "contains_html": 0.5,
            "in_blacklist": 0.5,
            "odd_sending_hours": 0,
            "link_count": 0
        }

        # common spam words
        mail_freq = email.word_frequencies_in_body()
        params["relative_word_freq"] = 0
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
        
        # number of words in body
        params["mail_length"] = nn_utils.sigmoid(len(email.body) / 200 - 1)

        # number of links in body
        params["link_count"] = nn_utils.sigmoid(int(email.link_count) - 3)

        # html tags in body
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
            self.rel_freq[i] = (freq_spam[i]-freq_ham[i]) / (freq_spam[i]+freq_ham[i])
        

class Email:
    def __init__(self, content) -> None:
        self.sender = None
        self.subject = None
        self.contains_html = False
        self.body = ""
        self.parse_email(content)

    def parse_email(self, content):
        email_object = email_lib.message_from_string(content)
        self.sender = email_object['from']
        self.subject = email_object['subject']

        if email_object['date']:
            time_xyz = email_object['date'].split(":")
            h = time_xyz[0][-2:]
            self.time = int(h)
        else:
            self.time = 12
            
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

        if utils.contains_html(self.body):
            self.contains_html = True
            self.body = re.sub(r"<[^>]*>", " ", self.body)
            self.body = re.sub(r"&nbsp", " ", self.body)

        self.link_count = utils.count_links(self.body)

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

