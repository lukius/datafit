import scipy
import math
import warnings

from score import BICScore


class DataClassifier(object):

    # Characterizes a data set assigning scores to every probability
    # distribution provided in scipy.stats. After adjusting the curves through
    # the maximum likelihood estimator, a score is computed (default is the
    # Bayesian Information Criterion, BIC). Distributions are then sorted by
    # this value --the lower, the better.
    
    def __init__(self, data):
        self.data = data

    def classify(self, score=None):
        warnings.filterwarnings("ignore")
        score = score or BICScore
        score_vals = list()
        
        for dist_str in dir(scipy.stats):
            if dist_str[0] == '_':
                continue
            try:
                dist = getattr(scipy.stats, dist_str)
                value = score(self.data, dist).value()
            except Exception:
                continue
            if math.isnan(value) or math.isinf(value):
                continue
            score_vals.append((dist_str, value))
            
        score_vals = sorted(score_vals, key=lambda t: t[1], cmp=score.cmp)
        
        return score_vals