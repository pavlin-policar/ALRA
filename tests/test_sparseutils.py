import unittest
import numpy as np
import scipy.sparse as sp

from ALRA.sparseutils import nonzero_mean, nonzero_std


class TestNonzeroMean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.array(
            [
                [0, 0, 19, 16, 11],
                [22, 17, 24, 23, 14],
                [0, 10, 13, 0, 0],
                [18, 0, 0, 0, 20],
                [0, 0, 21, 15, 12],
            ]
        )

    def test_dense_nonzero_mean_columns(self):
        expected = [20, 13.5, 19.25, 18, 14.25]
        computed = nonzero_mean(self.x, axis=0)
        np.testing.assert_almost_equal(computed, expected)

    def test_dense_nonzero_mean_rows(self):
        expected = [15.33333333, 20, 11.5, 19, 16]
        computed = nonzero_mean(self.x, axis=1)
        np.testing.assert_almost_equal(computed, expected)

    def test_sparse_nonzero_mean_columns(self):
        expected = [20, 13.5, 19.25, 18, 14.25]
        computed = nonzero_mean(sp.csr_matrix(self.x), axis=0)
        np.testing.assert_almost_equal(computed, expected)

    def test_sparse_nonzero_mean_rows(self):
        expected = [15.33333333, 20, 11.5, 19, 16]
        computed = nonzero_mean(sp.csr_matrix(self.x), axis=1)
        np.testing.assert_almost_equal(computed, expected)


class TestNonzeroStd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.array(
            [
                [0, 0, 19, 16, 11],
                [22, 17, 24, 23, 14],
                [0, 10, 13, 0, 0],
                [18, 0, 0, 0, 20],
                [0, 0, 21, 15, 12],
            ]
        )

    def test_dense_nonzero_std_columns(self):
        expected = [2.82842712, 4.94974746, 4.64578662, 4.35889894, 4.03112887]
        computed = nonzero_std(self.x, axis=0, ddof=1)
        np.testing.assert_almost_equal(computed, expected)

    def test_dense_nonzero_std_rows(self):
        expected = [4.04145188, 4.30116263, 2.12132034, 1.41421356, 4.58257569]
        computed = nonzero_std(self.x, axis=1, ddof=1)
        np.testing.assert_almost_equal(computed, expected)

    def test_sparse_nonzero_std_columns(self):
        expected = [2.82842712, 4.94974746, 4.64578662, 4.35889894, 4.03112887]
        computed = nonzero_std(sp.csr_matrix(self.x), axis=0, ddof=1)
        np.testing.assert_almost_equal(computed, expected)

    def test_sparse_nonzero_std_rows(self):
        expected = [4.04145188, 4.30116263, 2.12132034, 1.41421356, 4.58257569]
        computed = nonzero_std(sp.csr_matrix(self.x), axis=1, ddof=1)
        np.testing.assert_almost_equal(computed, expected)

    def test_sparse_doesnt_change_original_matrix(self):
        x = sp.csr_matrix(self.x)
        copy = x.copy()

        nonzero_std(x)

        np.testing.assert_almost_equal(x.toarray(), copy.toarray())
