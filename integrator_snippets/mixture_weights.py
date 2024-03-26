import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp


class MixtureWeights:

    def __init__(self):
        """Abstract class for mixture weights."""
        pass

    def log_weights(self) -> npt.NDArray[float]:
        """Log of the mixture weights.

        Parameters
        ----------
        :return: Log of the weights for each mixture component
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def update(self, attributes: dict):
        """Updates mixture weights. This should be used when one learns the weights, such as using an attention
        mechanism, see PhD thesis."""
        raise NotImplementedError


class UniformMixtureWeights(MixtureWeights):

    def __init__(self, T: int):
        """Uniform mixture weights, for all components.

        Parameters
        ----------
        :param T: Number of mixture components
        :type T: int

        Notes
        -----
        The weight of each component is 1 / (T+1).
        """
        super().__init__()
        self.T = T

    def log_weights(self) -> npt.NDArray[float]:
        """Computes log weights based on current T.

        Parameters
        ----------
        :return: Log of the weights for each mixture component
        :rtype: np.ndarray
        """
        return - np.log([self.T+1]*(self.T + 1))


class LinearMixtureWeights(MixtureWeights):

    def __init__(self, T: int, increasing: bool = True):
        """Mixture weights that are increasing (linearly) with the index of the component.

        Parameters
        ----------
        :param T: Number of mixture components
        :type T: int

        Notes
        -----
        The weight of each component is 1 / (T - k + 1), for k = 0, ..., T.
        """
        super().__init__()
        self.T = T
        self.increasing = increasing

    def log_weights(self) -> npt.NDArray[float]:
        """Computes log weights based on current T.

        Parameters
        ----------
        :return: Log of the weights for each mixture component
        :rtype: np.ndarray
        """
        if self.increasing:
            log_weights = - np.log(self.T - np.arange(self.T + 1) + 1)
        else:
            log_weights = - np.log(np.arange(1, self.T + 2))
        return log_weights - logsumexp(log_weights)


class MHMixtureWeights(MixtureWeights):

    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def log_weights(self) -> npt.NDArray[float]:
        """Computes log weights based on current T.

        Parameters
        ----------
        :return: Log of the weights for each mixture component
        :rtype: np.ndarray
        """
        # Metropolis-Hastings-like weights, uniform probability to k=0 and k=T, but everything else to 0
        log_weights = np.zeros(self.T + 1)
        log_weights[[0, -1]] = -np.log(2)
        return log_weights
