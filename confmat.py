class BinaryConfusionMatrix:

    def __init__(self, pos_tag, neg_tag) -> None:
        self.pos_tag = pos_tag
        self.neg_tag = neg_tag
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def as_dict(self):
        dictionary = {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn
        }
        return dictionary
    
    def update(self, truth, prediction):
        tags = (self.pos_tag, self.neg_tag)
        if(truth not in tags or prediction not in tags):
            raise ValueError
        if(prediction == self.pos_tag):
            if prediction == truth:
                self.tp += 1
            else:
                self.fp += 1
        else:
            if prediction == truth:
                self.tn += 1
            else:
                self.fn += 1
        

    def compute_from_dicts(self, truth_dict, pred_dict):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        for key in truth_dict:
            if pred_dict[key] == self.pos_tag:
                if truth_dict[key] == pred_dict[key]:
                    self.tp += 1
                else:
                    self.fp += 1
            else:
                if truth_dict[key] == pred_dict[key]:
                    self.tn += 1
                else:
                    self.fn += 1
