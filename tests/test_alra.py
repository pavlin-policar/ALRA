import unittest

import numpy as np
import scipy.sparse as sp
from os.path import join, abspath, dirname

from ALRA import ALRA, choose_k


class TestChooseK(unittest.TestCase):
    def test_dense(self):
        fname = join(dirname(abspath(__file__)), "yan.csv.gz")
        x = np.genfromtxt(fname, delimiter=",", skip_header=1)

        # Compute the true k with the original R implementation
        true_k = 13
        chosen_k = choose_k(x, k=30, noise_start=20)

        self.assertEqual(chosen_k, true_k)

    def test_sparse(self):
        fname = join(dirname(abspath(__file__)), "yan.csv.gz")
        x = np.genfromtxt(fname, delimiter=",", skip_header=1)
        x = sp.csr_matrix(x)

        # Compute the true k with the original R implementation
        true_k = 13
        chosen_k = choose_k(x, k=30, noise_start=20)

        self.assertEqual(chosen_k, true_k)


class TestALRA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # An arbitrary random array
        cls.x = np.array(
            [
                [0, 0, 19, 16, 11],
                [22, 17, 24, 23, 14],
                [0, 10, 13, 0, 0],
                [18, 0, 0, 0, 20],
                [0, 0, 21, 15, 12],
            ]
        )
        # Choose `k=2` because the imputation zeroes out entry (3, 2) = 10. This
        # way we can test that we properly restore this value
        cls.k = 2
        # Values were computed using the official R implementation by KlugerLab
        cls.correct = np.array(
            [
                [17.5304, 11.9893, 19.8301, 18.0959, 10.8308],
                [22.1777, 20.5596, 23.9098, 22.9704, 19.1215],
                [0.0, 10.0, 12.8241, 12.3482, 0.0],
                [22.7068, 8.9957, 0.0, 0.0, 16.001],
                [17.5851, 12.4553, 20.436, 18.5854, 11.0467],
            ]
        )

    def test_that_originaly_nonzero_values_remain_nonzero(self):
        """Make sure we didn't mistakenly zero out any original expression values."""
        x = self.x
        result = ALRA(x, k=self.k)

        self.assertFalse(np.any((result == 0) & (x > 0)))

    def test_dense_matrix(self):
        result = ALRA(self.x, k=self.k)
        np.testing.assert_almost_equal(result, self.correct, decimal=4)

    def test_sparse_matrix(self):
        x = sp.csr_matrix(self.x)
        result = ALRA(x, k=2)
        np.testing.assert_almost_equal(result, self.correct, decimal=4)
