from utils import read_classification_from_file
from confmat import BinaryConfusionMatrix
import os

def quality_score(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + 10*fp + fn)

def compute_quality_for_corpus(corpus_dir):
    truth = read_classification_from_file(os.path.join(corpus_dir, '!truth.txt'))
    prediction = read_classification_from_file(os.path.join(corpus_dir, '!prediction.txt'))
    stats = BinaryConfusionMatrix("SPAM", "HAM")
    stats.compute_from_dicts(truth, prediction)
    d = stats.as_dict()
    return quality_score(d["tp"], d["tn"], d["fp"], d["fn"])

# print(compute_quality_for_corpus(os.path.join(os.getcwd(), '1')))