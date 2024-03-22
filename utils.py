import numpy as np
from typing import Optional
import numpy.typing as npt
from scipy.special import logsumexp


def setup_rng(seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Sets up the random number generator with a given random seed, or it generates one.

    Parameters
    ----------
    :param seed: Random seed for the random number generator
    :type seed: int
    :param rng: Random number generator
    :type rng: np.random.Generator
    :return: Random number generator
    :rtype: np.random.Generator
    """
    assert (seed is None) or (rng is None), "At least one of `seed` or `rng` must be None."
    if rng is None:
        if seed is None:
            seed = np.random.randint(low=1000, high=9999)
        rng = np.random.default_rng(seed=seed)
    return rng


def uniform_log_kernel(ys: npt.NDArray[float], epsilon: float) -> npt.NDArray[float]:
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


def normal_log_kernel(ys: npt.NDArray[float], epsilon: float) -> npt.NDArray[float]:
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


def grad_neg_normal_log_kernel(xs: npt.NDArray[float], epsilon: float, jac: callable, f: callable)\
        -> npt.NDArray[float]:
    """Computes the gradient of negative log (normal) kernel. Used for HMC testing.

    Parameters
    ----------
    :param xs: (N, m) Matrix of points in \\mathbb{R}^m at which we with to compute the kernel
    :type xs: np.ndarray
    :param epsilon: Tolerance for the kernel
    :type epsilon: float
    :param jac: Jacobian of the function f
    :type jac: callable
    :param f: Function f
    :type f: callable
    :return: Gradient of the negative log kernel evaluated at each row of y with tolerance epsilon, has shape (N, m)
    :rtype: np.ndarray
    """
    return 0.5*np.einsum('ijk,ik->ik', jac(xs), f(xs)) / (epsilon**2)


def project(xs: npt.NDArray[float], vs: npt.NDArray[float], jac: callable) -> npt.NDArray[float]:
    """Projects a vector vs onto the normal space at xs.

    Parameters
    ----------
    :param xs: (N, m) Matrix of positions in \\mathbb{R}^m at which we with to project
    :type xs: np.ndarray
    :param vs: (N, m) Matrix of velocities in \\mathbb{R}^m to project
    :type vs: np.ndarray
    :param jac: Jacobian of the function f
    :type jac: callable
    :return: Projected velocities, has shape (N, m)
    :rtype: np.ndarray
    """
    qs = np.linalg.qr(np.transpose(jac(xs), axes=(0, 2, 1)), mode='reduced')[0]
    return np.einsum(
        'ijk,ik->ij',
        qs,
        np.einsum('ijk,ik->ij', np.transpose(qs, axes=(0, 2, 1)), vs)
    )


def essl(logw: npt.NDArray[float]) -> float:
    """ESS (Effective sample size) computed from log-weights.

    Parameters
    ----------
    :param logw: (N, ) log weights
    :type logw: np.ndarray
    :return: ESS of weights w = exp(lw), i.e. the quantity sum(w**2) / (sum(w))**2
    :rtype: float
    """
    return np.exp(2*logsumexp(logw) - logsumexp(2*logw))

