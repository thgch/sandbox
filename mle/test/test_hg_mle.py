"""
test_hg_mle.py

Author: Tomohide Higuchi
Created on 2020/05/14

Copyright 2020

"""

import unittest
import numpy as np
import hg_mle as testee


class TestMle(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # From here, test for each function #
    def test_log_likelihood(self):
        n, mu, sigma = 5, 2.0, 1.3
        samples = [0.8942807300000000000,
                   0.5538876500000000000,
                   1.9695362600000000000,
                   2.3317984700000000000,
                   1.9860110800000000000]
        ml_estimator = testee.Normal1DMle(samples)
        log_likelihood = ml_estimator.log_likelihood(np.array([mu, sigma]))
        self.assertAlmostEqual(log_likelihood, -6.91984830401281)

    def test_estimate_params(self):
        # n, mu, sigma = 5, 2.0, 1.3
        samples = [0.8942807300000000000,
                   0.5538876500000000000,
                   1.9695362600000000000,
                   2.3317984700000000000,
                   1.9860110800000000000]
        ml_estimator = testee.Normal1DMle(samples)
        ml_estimator.estimate_params()
        actual = ml_estimator.params
        expected = np.array([1.54710219347038000, 0.69274708864698500])
        np.testing.assert_almost_equal(actual, expected, decimal=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)