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
        cls.x1 = np.array([
            [ 0,  0, 19, 16, 11],
            [22, 17, 24, 23, 14],
            [ 0, 10, 13,  0,  0],
            [18,  0,  0,  0, 20],
            [ 0,  0, 21, 15, 12],
        ])
        # This one tests a non-square matrix
        cls.x2 = np.array([
            [ 0,  0, 19, 16, 11,  0,  0],
            [22, 17, 24, 23, 14,  0,  1],
            [ 0, 10, 13,  0,  0,  5,  2],
            [18,  0,  0,  0, 20, 32,  0],
            [ 0,  0, 21, 15, 12,  0, 21],
        ])
        # This one tests the case where we have zero-only columns
        cls.x3 = np.array([
            [ 0,  0, 19, 16, 11,  0,  0],
            [22, 17, 24, 23, 14,  0,  0],
            [ 0, 10, 13,  0,  0,  0,  0],
            [18,  0,  0,  0, 20,  0,  0],
            [ 0,  0, 21, 15, 12,  0,  0],
        ])
        # Choose `k=2` because the imputation zeroes out entry (3, 2) = 10. This
        # way we can test that we properly restore this value
        cls.k = 2
        # Values were computed using the official R implementation by KlugerLab
        cls.correct1 = np.array([
            [17.5304, 11.9893, 19.8301, 18.0959, 10.8308],
            [22.1777, 20.5596, 23.9098, 22.9704, 19.1215],
            [ 0.0000, 10.0000, 12.8241, 12.3482,  0.0000],
            [22.7068,  8.9957,  0.0000,  0.0000, 16.0010],
            [17.5851, 12.4553, 20.4360, 18.5854, 11.0467],
        ])
        cls.correct2 = np.array([
            [17.6790, 12.5813, 18.5872, 17.3615, 10.4220,  0.0000,  0.0000],
            [20.8954, 19.2862, 24.2415, 22.7212, 17.0917,  5.0000, 18.9110],
            [ 0.0000,  7.3697, 13.2109, 12.3553,  0.0000,  5.0000,  2.0000],
            [23.6106,  0.0000,  0.0000,  0.0000, 18.3163, 32.0000,  0.0000],
            [17.8149, 14.7628, 20.9603, 19.5620, 11.1701,  0.0000,  8.6856],
        ])
        cls.correct3 = np.array([
            [17.5304, 11.9893, 19.8301, 18.0959, 10.8308,  0.0000,  0.0000],
            [22.1777, 20.5596, 23.9098, 22.9704, 19.1215,  0.0000,  0.0000],
            [ 0.0000, 10.0000, 12.8241, 12.3482,  0.0000,  0.0000,  0.0000],
            [22.7068,  8.9957,  0.0000,  0.0000, 16.0010,  0.0000,  0.0000],
            [17.5851, 12.4553, 20.4360, 18.5854, 11.0467,  0.0000,  0.0000],
        ])

    def test_that_originaly_nonzero_values_remain_nonzero(self):
        """Make sure we didn't mistakenly zero out any original expression values."""
        x = self.x1
        result = ALRA(x, k=self.k)

        self.assertFalse(np.any((result == 0) & (x > 0)))

    def test_dense_matrix_1(self):
        result = ALRA(self.x1, k=self.k)
        np.testing.assert_almost_equal(result, self.correct1, decimal=4)

    def test_sparse_matrix_1(self):
        x = sp.csr_matrix(self.x1)
        result = ALRA(x, k=2)
        np.testing.assert_almost_equal(result, self.correct1, decimal=4)

    def test_dense_matrix_2(self):
        result = ALRA(self.x2, k=self.k)
        np.testing.assert_almost_equal(result, self.correct2, decimal=4)

    def test_sparse_matrix_2(self):
        x = sp.csr_matrix(self.x2)
        result = ALRA(x, k=2)
        np.testing.assert_almost_equal(result, self.correct2, decimal=4)

    def test_dense_matrix_3(self):
        result = ALRA(self.x3, k=self.k)
        np.testing.assert_almost_equal(result, self.correct3, decimal=4)

    def test_sparse_matrix_3(self):
        x = sp.csr_matrix(self.x3)
        result = ALRA(x, k=2)
        np.testing.assert_almost_equal(result, self.correct3, decimal=4)
