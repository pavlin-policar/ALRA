import logging
import numpy as np
from fbpca import pca
from scipy.stats import norm

from .sparseutils import nonzero_mean, nonzero_std, find_zeroed_indices

log = logging.getLogger("ALRA")


def choose_k(X, k=100, pval_thresh=1e-10, noise_start=80, n_iter=2):
    if k > min(X.shape):
        raise ValueError(
            f"`k` must be smaller than `min(N, M)`. Maximum value "
            f"can be {min(X.shape)} but `{k}` given"
        )

    if noise_start > k - 5:
        raise ValueError("At least 5 singular values must be considered noise.")

    U, s, Va = pca(X, k=k, n_iter=n_iter, raw=True)

    differences = np.diff(s)

    mean = np.mean(differences[noise_start - 1 :])
    std = np.std(differences[noise_start - 1 :], ddof=1)

    probabilities = norm.pdf(differences, loc=mean, scale=std)

    k = np.max(np.argwhere(probabilities < pval_thresh)) + 1

    return k


def ALRA(X, k=None, n_iter=10):
    """Adaptively-thresholded Low Rank Approximation.

    Parameters
    ----------
    X: array_like
    k: int
    n_iter: int

    Returns
    -------
    np.array

    """
    if k is None:
        k = choose_k(X)
        log.info(f"No `k` given. Automatically determined `k={k}`.")

    # Compute the SVD and compute the rank-k reconstruction
    U, s, Va = pca(X, k=k, n_iter=n_iter, raw=True)
    X_rank_k = U * s @ Va

    X_rank_k = np.ma.masked_array(X_rank_k)

    # Find the absolute values of the minimum expression levels for each gene
    minimum_expressions = np.abs(np.min(X_rank_k, axis=0))
    # Zero out all expressions with values below the gene minimum value
    X_rank_k[X_rank_k <= minimum_expressions] = np.ma.masked

    # Rescale the expressions so the first two moments match the original matrix
    X_mean, X_std = nonzero_mean(X, axis=0), nonzero_std(X, axis=0, ddof=1)
    X_rk_mean, X_rk_std = X_rank_k.mean(axis=0), X_rank_k.std(axis=0, ddof=1)

    scale = X_std / X_rk_std
    translate = -X_rk_mean * scale + X_mean

    scale_columns = ~np.isnan(X_std) & ~np.isnan(X_rk_std)
    X_rank_k[:, scale_columns] *= scale[scale_columns]
    X_rank_k[:, scale_columns] += translate[scale_columns]

    # Values can become negative during rescaling, so we zero those out
    X_rank_k[X_rank_k < 0] = np.ma.masked

    # Restore potentially zeroed out expression values which appeared in the
    # original expression matrix. Where both values are non-zero, prefer the
    # rank-k approximation
    zeroed_out_indices = find_zeroed_indices(X_rank_k, X)
    X_rank_k[zeroed_out_indices] = X[zeroed_out_indices]

    log.info(
        f"{len(zeroed_out_indices[0])} original expression values were "
        f"zeroed out during imputation and restored to original values."
    )

    X_rank_k = X_rank_k.filled(0)

    return X_rank_k
