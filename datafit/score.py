import scipy

import numpy as np

from statsmodels.base.model import GenericLikelihoodModel


class FitScore(object):

    @classmethod
    def cmp(cls, a, b):
        return -1 if a > b else 1
    
    def __init__(self, data, dist):
        self.data = data
        self.dist = dist
        self._fit()
    
    def _fit(self):
        params = self.dist.fit(self.data)
        
        self.fitter = GenericLikelihoodModel(self.data)
        self.fitter.nparams = len(params)
        self.fitter.loglike = self._loglike
        
        self.fit_results = self.fitter.fit(start_params=params,
                                           disp=False,
                                           maxiter=2000)
        
    def _loglike(self, params):
        return self.dist.logpdf(self.fitter.endog, *params).sum()
    
    def value(self):
        raise NotImplementedError


class InformationCriterion(FitScore):
    
    @classmethod
    def cmp(cls, a, b):
        return -1 if a < b else 1
    

class BICScore(InformationCriterion):
    
    def value(self):
        return -2. * self.fit_results.llf +\
               np.log(self.fit_results.nobs) * len(self.fit_results.params)


class AICScore(InformationCriterion):
    
    def value(self):
        return -2. * self.fit_results.llf +\
                2 * len(self.fit_results.params)    


class KSTestScore(FitScore):
    
    def value(self):
        _, pval = scipy.stats.kstest(rvs=self.data,
                                     cdf=self.dist.name,
                                     args=self.fit_results.params)
        return pval