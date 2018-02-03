import scipy
import math
import warnings

import numpy as np

from statsmodels.base.model import GenericLikelihoodModel


class DataClassifier(object):

    # Characterizes a data set assigning scores to every probability
    # distribution provided in scipy.stats. After adjusting the curves through
    # the maximum likelihood estimator, the Bayesian Information Criterion
    # (BIC) score is computed. Distributions are then sorted by this value
    # --the lower, the better.

    def classify(self, data):
        warnings.filterwarnings("ignore")
        scores = list()
        for dist in dir(scipy.stats):
            if dist[0] == '_':
                continue
            try:
                score = DistributionFitter(dist).fit(data)
            except Exception:
                continue
            if math.isnan(score) or math.isinf(score) or score < 0:
                continue
            scores.append((dist, score))
        scores = sorted(scores, key=lambda t: t[1])
        return scores
                    

class DistributionFitter(object):

    # Given a data set and the name of a probability distribution in
    # scipy.stats, this object adjusts by maximum likelihood this 
    # distribution to the data and computes the BIC score.

    def __init__(self, dist):
        self.distribution = getattr(scipy.stats, dist)

    def loglike(self, params):
        return self.distribution.logpdf(self.fitter.endog, *params).sum()

    def fit(self, data):
        params = self.distribution.fit(data)
        self.fitter = GenericLikelihoodModel(data)
        self.fitter.nparams = len(params)
        self.fitter.loglike = self.loglike
        results = self.fitter.fit(start_params=params, disp=False, maxiter=2000)
        return self.BIC(results, params)

    def BIC(self, results, params):
        return -2. * results.llf + np.log(results.nobs) * len(params) 
