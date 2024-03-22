import numpy as np
from typing import Optional
import numpy.typing as npt
import scipy as sp
from scipy.special import logsumexp


def setup_rng(seed: Optional[int] = None, rng: Optional[np.random.Generator] = None):
    """Sets up the random number generator with a given random seed, or it generates one."""
    assert (seed is None) or (rng is None), "At least one of `seed` or `rng` must be None."
    if rng is None:
        if seed is None:
            seed = np.random.randint(low=1000, high=9999)
        rng = np.random.default_rng(seed=seed)
    return rng


def uniform_log_kernel(ys: npt.NDArray[float], epsilon: float):
    """Vectorised uniform log-kernel.

    Parameters
    ----------
    :param ys: (N, m) Matrix of points in \\mathbb{R}^m at which we with to compute the kernel
    :type ys: np.ndarray
    :param epsilon: Tolerance for the kernel
    :type epsilon: float
    :return: Log kernel evaluated at each row of y with bandwidth/tolerance epsilon, has shape (N, )
    :rtype: np.ndarray
    """
    assert isinstance(epsilon, (int, float)), "Epsilon must be a float."
    assert epsilon > 0.0, "Epsilon must be larger than zero."
    log_kernel_values = np.full(len(ys), -np.inf)
    active = (np.linalg.norm(ys, axis=1) <= epsilon)
    log_kernel_values[active] = -ys.shape[1]*np.log(epsilon)
    return log_kernel_values


def normal_log_kernel(ys: npt.NDArray[float], epsilon: float):
    """Vectorised normal log kernel.

     Parameters
     ----------
    :param ys: (N, m) Matrix of points in \\mathbb{R}^m at which we with to compute the kernel
    :type ys: np.ndarray
    :param epsilon: Tolerance for the kernel
    :type epsilon: float
    :return: Log kernel evaluated at each row of y with bandwidth/tolerance epsilon, has shape (N, )
    :rtype: np.ndarray
     """
    assert isinstance(epsilon, (int, float)), "Epsilon must be a float."
    assert epsilon > 0.0, "Epsilon must be larger than zero."
    return - (np.linalg.norm(ys, axis=1)**2) / (2*(epsilon**2)) - ys.shape[1]*np.log(epsilon)


def grad_neg_normal_log_kernel(xs: npt.NDArray[float], epsilon: float, jac: callable, f: callable):
    """Computes the gradient of negative log (normal) kernel. Used for HMC."""
    return 0.5*np.einsum('ijk,ik->ik', jac(xs), f(xs)) / (epsilon**2)


def project(xs: npt.NDArray[float], vs: npt.NDArray[float], jac: callable):
    qs = np.linalg.qr(np.transpose(jac(xs), axes=(0, 2, 1)), mode='reduced')[0]
    return np.einsum(
        'ijk,ik->ij',
        qs,
        np.einsum('ijk,ik->ij', np.transpose(qs, axes=(0, 2, 1)), vs)
    )


def essl(logw):
    """CHOPIN
    ESS (Effective sample size) computed from log-weights.

    Parameters
    ----------
    logw : (N, ) ndarray
        log-weights

    Returns
    -------
    float
        the ESS of weights w = exp(lw), i.e. the quantity
        sum(w**2) / (sum(w))**2

    Note
    ----
    The ESS is a popular criterion to determine how *uneven* are the weights.
    Its value is in the range [1, N], it equals N when weights are constant,
    and 1 if all weights but one are zero.

    """
    return np.exp(2*logsumexp(logw) - logsumexp(2*logw))

