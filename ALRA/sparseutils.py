import numpy as np
import scipy.sparse as sp


def nonzero_mean(X, axis=0):
    """Compute the mean of non-zero values in a given matrix.

    Parameters
    ----------
    X: array_like
    axis: int

    Returns
    -------
    np.ndarray

    """
    if sp.issparse(X):
        if axis == 0:
            X = X.tocsc()
        elif axis == 1:
            X = X.tocsr()
        else:
            raise NotImplementedError(
                f"`axis={axis}` is not implemented for " f"sparse matrices."
            )

        counts = np.diff(X.indptr)
        sums = X.sum(axis=axis)
        sums = np.asarray(sums).ravel()
        with np.errstate(invalid="ignore"):
            return sums / counts

    else:
        X = np.ma.masked_array(X, mask=X <= 0)
        means = X.mean(axis=axis)
        return means.filled(0)


def nonzero_var(X, axis=0, ddof=0):
    """Compute the variance of non-zero values in a given matrix.

    Parameters
    ----------
    X: array_like
    axis: int
    ddof: int

    Returns
    -------
    np.ndarray

    """
    if sp.issparse(X):
        # We'll modify X inplace so we need to create a copy
        if axis == 0:
            X = X.tocsc(copy=True)
        elif axis == 1:
            X = X.tocsr(copy=True)
        else:
            raise NotImplementedError(
                f"`axis={axis}` is not implemented " f"sparse matrices."
            )

        X = X.astype(float)

        means = nonzero_mean(X, axis=axis)
        i, j, v = sp.find(X)
        if axis == 0:
            X[i, j] = v - means[j]
        elif axis == 1:
            X[i, j] = v - means[i]

        X.data = X.data ** 2

        counts = np.diff(X.indptr) - ddof
        sums = X.sum(axis=axis)
        sums = np.asarray(sums).ravel()
        with np.errstate(invalid="ignore"):
            return sums / counts

    else:
        X = np.ma.masked_array(X, mask=X <= 0)
        variances = X.var(axis=axis, ddof=ddof)
        return variances.filled(0)


def nonzero_std(X, axis=0, ddof=0):
    """Compute the standard deviation of non-zero values in a given matrix.

    Parameters
    ----------
    X: array_like
    axis: int
    ddof: int

    Returns
    -------
    np.ndarray

    """
    return np.sqrt(nonzero_var(X, axis=axis, ddof=ddof))


def find_zeroed_indices(adjusted, original):
    """Find the indices of the values present in ``original`` but missing in ``adjusted``.

    Parameters
    ----------
    adjusted: np.array
    original: array_like

    Returns
    -------
    Tuple[np.ndarray]
        Indices of the values present in ``original`` but missing in ``adjusted``.

    """
    if sp.issparse(original):
        i, j, v = sp.find(original)
        # Use hash maps to figure out which indices have been lost in the original
        original_indices = set(zip(i, j))
        adjusted_indices = set(zip(*np.where(~adjusted.mask)))
        zeroed_indices = original_indices - adjusted_indices

        # Convert our hash map of coords into the standard numpy indices format
        indices = list(zip(*zeroed_indices))
        indices = tuple(map(np.array, indices))

        return indices

    else:
        original = np.ma.masked_array(original, mask=original <= 0)
        return np.where(adjusted.mask & ~original.mask)
