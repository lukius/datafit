#!/usr/bin/python2.7

import unittest

from test import test_classify


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_classify.ClassifierTest)
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    run_tests()