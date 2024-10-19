import numpy as np
import scipy as sp
import numpy.typing as npt
from .utils import uniform_log_kernel, normal_log_kernel, essl
from .manifolds import Manifold


class SequentialTargets:

    def __init__(self, param_init: float):
        """Abstract class for sequential targets. This is the base class for Filamentary and Tempered distributions.

        Parameters
        ----------
        :param param_init: Initial parameter for the initial distribution
        :type param_init: float
        """
        self.param_old = param_init
        self.param_new = param_init

    def update_parameter(self, attributes: dict):
        """Sets the old parameter to the current one, and updates the current one.

        Parameters
        ----------
        :param attributes: Attributes of the integrator snippet
        :type attributes: dict
        """
        self.param_old = self.param_new  # new becomes old
        self.param_new = self.generate_parameter(attributes)  # generate new parameter

    def generate_parameter(self, attributes: dict) -> float:
        """Generates a new parameter, e.g. a new epsilon or new tempering parameter.

        Parameters
        ----------
        :param attributes: Attributes of the integrator snippet
        :type attributes: dict
        :return: New parameter
        :rtype: float
        """
        raise NotImplementedError

    def terminate(self) -> bool:
        """Checks if the nth parameter is the terminal one.

        Parameters
        ----------
        :return: Whether we have reached the final distribution
        :rtype: bool
        """
        raise NotImplementedError

    def logw(self, pos_nk: npt.NDArray[float], aux_nk: npt.NDArray[float]) -> npt.NDArray[float]:
        """Computes log weights for each of the N(T+1) particles.

        Parameters
        ----------
        :param pos_nk: Positions of the particles, has shape (N, T+1, d)
        :type pos_nk: np.ndarray
        :param aux_nk: Auxiliary variables of respective particles, has shape (N, T+1, d)
        :type aux_nk: np.ndarray
        :return Log weights for each of the N(T+1) particles, has shape (N, T+1)
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def sample_initial_particles(self, N: int) -> npt.NDArray[float]:
        """Samples initial particles to initialise the integrator snippet. Should be implemented by user.

        Parameters
        ----------
        :param N: Number of particles to sample
        :type N: int
        :return: Initial particles, has shape (N, d)
        :rtype: np.ndarray
        """
        raise NotImplementedError


class Filamentary(SequentialTargets):

    def __init__(self, manifold: Manifold, eps: float, kernel: str = 'uniform', eps_ter: float = 1e-16, qv: float = 0.8,
                 coeff: float = 1.0):
        """Filamentary distributions concentrated around a lower-dimensional manifold.

        Parameters
        ----------
        :param manifold: Manifold around which the distribution is concentrated
        :type manifold: Manifold
        :param eps: Initial tolerance for the kernel, typically will be computed as maximum distance of the initial
        particles from the manifold
        :type eps: float
        :param kernel: Type of kernel to use, either 'uniform' or 'normal'
        :type kernel: str
        :param eps_ter: Terminal tolerance for the kernel
        :type eps_ter: float
        :param qv: The next tolerance will be set to the `qv` quantile of the distances of the particles from the
        manifold
        :type qv: float
        :param coeff: Coefficient to multiply the current tolerance by to get the new tolerance, should be used with
        a normal kernel to avoid the sequence of distributions getting stuck
        :type coeff: float

        Notes
        -----
        This implements $\\mu_n = \\pi_n \\otimes \\varpi_n$ where we assume that $\\varpi_n$ is fixed and does not
         change with n, and we assume $\\pi_n$ is a filamentary distribution on $(X, \\mathcal{X})$.
        """
        super().__init__(param_init=eps)
        assert kernel in {'uniform', 'normal'}, "Kernel must be either 'uniform' or 'normal'."
        self.kernel_type = kernel
        self.log_kernel = uniform_log_kernel if kernel == 'uniform' else normal_log_kernel
        self.eps_ter = eps_ter
        self.manifold = manifold
        self.qv = qv
        self.coeff = 1.0 if kernel == 'uniform' else coeff

    def generate_parameter(self, attributes: dict) -> float:
        """Generates the new tolerance using automatic tolerance selection strategy.

        Parameters
        ----------
        :param attributes: Attributes of the integrator snippet
        :type attributes: dict
        :return: New tolerance parameter
        :rtype: float
        """
        return max(
            min(
                float(self.coeff * self.param_old),
                np.quantile(np.unique(self.manifold.distances(attributes['pos'])), q=self.qv)
            ),
            self.eps_ter
        )

    def terminate(self) -> bool:
        """Simply checks if a certain epsilon is <= the terminal epsilon.

        Parameters
        ---------
        :return: Whether we have reached the final filamentary distribution
        :rtype: bool
        """
        return self.eps_ter > self.param_old or abs(self.param_old - self.eps_ter) <= 1e-10  # param would be epsilon

    def base_log_dens_x(self, xnk_flattened: npt.NDArray[float]) -> npt.NDArray[float]:
        """Unconstrained density on the marginal space $(X, \\mathcal{X})$.

        Parameters
        ----------
        :param xnk_flattened: Flattened array of positions, has shape (N*(T+1), d)
        :type xnk_flattened: np.ndarray
        :return: Log density evaluated at each row of xnk_flattened, has shape (N*(T+1), )
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def log_dens_aux(self, vnk_flattened: npt.NDArray[float]) -> npt.NDArray[float]:
        """Computes log density for auxiliary variables (assumed not to vary for n).

        Parameters
        ----------
        :param vnk_flattened: Flattened array of auxiliary variables, has shape (N*(T+1), d)
        :type vnk_flattened: np.ndarray
        :return: Should return the log density for the auxiliary variables of shape (N*(T+1), )
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def logw(self, pos_nk: npt.NDArray[float], aux_nk: npt.NDArray[float]) -> npt.NDArray[float]:
        """Computes log weights.

        Parameters
        ----------
        :param pos_nk: Positions of the particles, has shape (N, T+1, d)
        :type pos_nk: np.ndarray
        :param aux_nk: Auxiliary variables of respective particles, has shape (N, T+1, d)
        :type aux_nk: np.ndarray
        :return: Log weights for each of the N(T+1) particles, has shape (N, T+1)
        :rtype: np.ndarray
        """
        N, Tp1, d = pos_nk.shape  # (N, T+1, d)
        # Log-denominator for the un-normalised weights with shape (N, 1)
        log_den = self.base_log_dens_x(pos_nk[:, 0]) + self.log_kernel(self.manifold.f(pos_nk[:, 0]), self.param_old)
        log_den += self.log_dens_aux(aux_nk[:, 0])
        log_den = log_den[:, None]  # (N, 1)
        # Log-numerator for the un-normalized weights
        pos_nk = pos_nk.reshape(-1, d)  # (N*(T+1), d)
        aux_nk = aux_nk.reshape(-1, d)   # (N*(T+1), d)
        # Log-numerator for the un-normalised weights with shape (N, T)
        log_num = (self.base_log_dens_x(pos_nk) + self.log_kernel(self.manifold.f(pos_nk), self.param_new)
                   + self.log_dens_aux(aux_nk))
        log_num[np.isnan(log_num)] = -np.inf  # replace NaNs with -inf (for HMC)
        log_num = log_num.reshape(N, Tp1)
        return log_num - log_den  # (N, T+1)


class FilamentaryFixedTolSeq(Filamentary):

    def __init__(self, manifold: Manifold, epsilons: npt.NDArray[float], kernel='uniform'):
        """This class implements a sequence of filamentary distributions where the epsilons are predetermined and
        given by the user. The key is to always know the index of the current iteration.
        All methods utilize param_new and param_old. We just need to make sure that at any given moment, these
        are the correct epsilons.
        Initially both param_new and param_old are set to epsilons[0]. When we generate a new parameter, we need to
        increase the counter"""
        super().__init__(manifold=manifold, eps=float(epsilons[0]), kernel=kernel, eps_ter=float(epsilons[-1]))
        self.epsilons = epsilons
        self.n = 0

    def generate_parameter(self, attributes: dict) -> float:
        """Grabs the next tolerance from the fixed list."""
        self.n += 1
        return float(self.epsilons[self.n])


class Tempered(SequentialTargets):

    def __init__(self, gamma: float, data: dict, alpha: float = 0.9, tol: float = 1e-8):
        """Standard Tempering in SMC samplers. Suppose \\pi(x) is the prior and L(x) is the likelihood. Then the
        tempered target is

        ```math
            \\pi_n(x) \\propto \\pi(x) L(x)^{\\gamma_n}
        ```

        where \\gamma_n is the tempering parameter. We start at \\gamma_0 = 0.0 and end at \\gamma_P = 1.0. Notice that
        this is the tempered target on the x space. The auxiliary variables are velocities (since we are using a
        Hamiltonian integrator) and therefore the full tempered target is actually

        ```math
            \\mu_n(x) \\propto \\pi_n(x) \\otimes \\varpi_n(v)
        ```

        Suppose we have particle z = (x, v) and we generate particle z_k = (x_k, v_k) by applying the Leapfrog
        integrator to z for k times. Then, the integrator snippet log-weights are

        ```math
            \\log w_k = \\log\\pi(x_k) + \\gamma_n \\log L(x_k) - \\log\\varpi(v_k)
                         - \\log\\pi(x) - \\gamma_{n-1} \\log L(x) - \\log\\varpi(v)
        ```

        Importantly, the log-weights are a function of z, z_k, \\gamma_{n-1} and \\gamma_n. One can also simply say they
        are a function of z, z_k and "\\gamma_n - \\gamma_{n-1}". When doing adaptive tempering, we wish to find a new
        tempering parameter such that we retain a certain ESS. Of course z and z_k do not depend on the tempering
        parameters.

        """
        assert 0.0 <= gamma <= 1.0, "Tempering parameter must be in (0, 1]."
        assert 0.0 < alpha < 1.0, "Alpha must be in (0, 1)."
        super().__init__(param_init=gamma)
        self.data = data  # data for the likelihood
        self.alpha = alpha  # proportion of ESS that we wish to retain when using Brent's method
        self.tol = tol  # tolerance to check if we have reached the final distribution

    def log_prior(self, xs: npt.NDArray[float]):
        """Computes the prior for the untempered distribution."""
        raise NotImplementedError

    def log_likelihood(self, xs: npt.NDArray[float]):
        """Computes the likelihood for the untempered distribution."""
        raise NotImplementedError

    def log_dens_aux(self, vs: npt.NDArray[float]):
        """Computes the log density for the auxiliary variables."""
        raise NotImplementedError

    def logw(self, pos_nk: npt.NDArray[float], aux_nk: npt.NDArray[float], param_old: float = None,
             param_new: float = None):
        """Computes log weights for znk = [xnk, vnk]. Here we DO expect shape (N, T+1, d) for each array.
        Notice that for tempered distributions mu_n = pi_n otimes varpi and varpi is fixed. Here pi_n is tempered,
        so it has the form pi_n = pi L(x)^gamma_n."""
        assert (param_old is None and param_new is None) or (param_old is not None and param_new is not None), \
            "Either both parameters are provided, or none."
        if param_old is None and param_new is None:
            param_old = self.param_old
            param_new = self.param_new
        N, Tp1, d = pos_nk.shape
        # Log denominator for the un-normalised weights with shape (N, 1)
        log_den = self.log_prior(pos_nk[:, 0]) + param_old * self.log_likelihood(pos_nk[:, 0])
        log_den += self.log_dens_aux(aux_nk[:, 0])
        log_den = log_den[:, None]  # (N, 1)
        # Log numerator for the un-normalised weights (N, T)
        pos_nk = pos_nk.reshape(-1, d)  # (N*(T+1), d)
        aux_nk = aux_nk.reshape(-1, d)   # (N*(T+1), d)
        # Log-numerator for the un-normalised weights with shape (N, T)
        log_num = self.log_prior(pos_nk) + param_new * self.log_likelihood(pos_nk) + self.log_dens_aux(aux_nk)
        log_num = log_num.reshape(N, Tp1)
        return log_num - log_den

    def generate_parameter(self, attributes: dict):
        """Grabbed from Chopin's particles package. Uses Brent's method to find next parameter.
        We use the ESS for mu bar."""
        N = attributes['N']

        def f(e):
            ess = essl(e * attributes['logw_folded']) if e > 0.0 else N  # avoid 0 * inf issue when e == 0
            return ess - self.alpha * N
        if f(1. - self.param_new) < 0.:
            root, res = sp.optimize.brentq(f, 0.0, 1.0 - self.param_new, full_output=True)
            return self.param_new + root
        else:
            return 1.0

    def terminate(self):
        """Tempered distributions end when gamma=1.0"""
        return abs(self.param_new - 1.0) <= self.tol
