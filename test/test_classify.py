import unittest
import scipy.stats

import numpy as np

from datafit.classifier import DataClassifier
from datafit.score import BICScore, AICScore, KSTestScore


class ClassifierTest(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        
        uniform_sample = np.linspace(-20, 20, 30)
        self.unif_classifier = DataClassifier(uniform_sample)
        
        bradford_sample = scipy.stats.bradford.rvs(4.32, size=300)
        self.brad_classifier = DataClassifier(bradford_sample)
        
    def _test_best_distribution(self, results, target):
        self.assertGreater(len(results), 0)
        
        # Actual score is really not needed here.
        best_dist, _ = results[0]
        self.assertEqual(best_dist, target)
        
    def _test_distribution_in_top_5(self, results, target):
        self.assertGreater(len(results), 5)
        
        for dist, _ in results[:5]:
            if dist == target:
                return
            
        self.fail('%s not found in top 5 distributions' % target)
    
    def test_uniform_in_top_5_with_BIC_score(self):
        results = self.unif_classifier.classify(score=BICScore)
        self._test_distribution_in_top_5(results, 'uniform')
        
    def test_uniform_in_top_5_with_AIC_score(self):
        results = self.unif_classifier.classify(score=AICScore)
        self._test_distribution_in_top_5(results, 'uniform')
        
    def test_uniform_in_top_5_with_KS_test_score(self):
        results = self.unif_classifier.classify(score=KSTestScore)
        self._test_distribution_in_top_5(results, 'uniform')
        
    def test_bradford_sample_and_KS_test_score(self):
        results = self.brad_classifier.classify(score=KSTestScore)
        self._test_best_distribution(results, 'bradford')