import numpy as np
import numpy.typing as npt
from typing import Optional
from scipy.linalg import cho_factor, cho_solve
from distributions import SequentialTargets, Filamentary
from utils import project


class Integrator:

    def __init__(self, x_dim: int, v_dim: int, T: int):
        """Abstract integrator class.

        Parameters
        ----------
        :param x_dim: Dimensionality of the position space, the actual variable of interest
        :type x_dim: int
        :param v_dim: Dimensionality of the auxiliary space, the variables that help us move in the position space
        :type v_dim: int
        """
        self.x_dim = x_dim  # dimension of position variable
        self.v_dim = v_dim  # dimension of auxiliary variables
        self.T = T  # number of integration steps

    def integrate(self, xs: npt.NDArray[float], vs: npt.NDArray[any], target: SequentialTargets) \
            -> tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Creates trajectories.

        Parameters
        ----------
        :param xs: (N, x_dim) array of positions
        :type xs: np.ndarray
        :param vs: (N, v_dim) array of auxiliary variables
        :type vs: np.ndarray
        :param target: Target distribution
        :type target: SequentialTargets
        :return: Tuple of arrays (positions, velocities) of shape (N, T+1, x_dim) and (N, T+1, v_dim)
        :rtype: tuple
        """
        raise NotImplementedError("integrate method not implemented.")

    def sample_auxiliaries(self, N: int, rng: Optional[np.random.Generator] = None) -> npt.NDArray[any]:
        """Samples auxiliary variables used by the integrator.

        Parameters
        ----------
        :param N: Number of particles, and thus of samples to be drawn
        :type N: int
        :param rng: Random number generator for reproducibility
        :rtype rng: np.random.Generator
        :return: Sampled auxiliary variables of shape (N, v_dim)
        :rtype: np.ndarray
        """
        raise NotImplementedError("sample_auxiliaries method not implemented.")

    def eval_aux_logdens(self, vs: npt.NDArray[any]) -> npt.NDArray[float]:
        """Evaluates the log of the density of the auxiliary variables.

        Parameters
        ----------
        :param vs: Auxiliary variables at which we evaluate the log density, has shape (N, v_dim)
        :type vs: np.ndarray
        :return: Log density of the auxiliary variables of shape (N, )
        :rtype: np.ndarray
        """
        raise NotImplementedError("eval_logdens method not implemented.")

    def update_params(self, **kwargs):
        """Updates attributes based on Stochastic Approximation of IntegratorSnippets.

        Parameters
        ----------
        :param kwargs: Dictionary of attributes to update
        :type kwargs: dict
        """
        raise NotImplementedError("update_params method not implemented.")


class LeapfrogIntegrator(Integrator):

    def __init__(self, d: int, T: int, step_size: float, dVdq: callable,
                 mass_matrix: Optional[npt.NDArray[float]] = None):
        """Classic Leapfrog Integrator used to integrate Hamilton's equations, with constant mass matrix.

        Parameters
        ----------
        :param d: Dimensionality of the position space
        :type d: int
        :param T: Number of leapfrog steps
        :type T: int
        :param step_size: Step size of the integrator
        :type step_size: float
        :param dVdq: Gradient of the potential, must take arrays (N, d) as input
        :type dVdq: callable
        :param mass_matrix: Mass matrix of the system, defaults to identity
        :type mass_matrix: np.ndarray
        """
        super().__init__(x_dim=d, v_dim=d, T=T)
        self.step_size = step_size
        self.mass_matrix = mass_matrix if mass_matrix is not None else np.eye(self.x_dim)
        self.chol_mass, _ = cho_factor(self.mass_matrix, lower=True)
        self.aux_nc = -(d / 2) * np.log(2 * np.pi) - np.log(np.diag(self.chol_mass)).sum()  # norm const of aux density
        self.rng = np.random.default_rng(np.random.randint(low=1000, high=9999))
        self.dVdq = dVdq  # gradient of the potential, must work for a whole array (N, d)
        if np.allclose(self.mass_matrix, np.eye(self.x_dim)):
            self.v_transform = lambda vs: vs
        else:
            self.v_transform = lambda vs: cho_solve((self.chol_mass, True), vs.T).T

    def update_params(self, **kwargs):
        """Updates step size and number of integrator steps.

        Parameters
        ----------
        :param kwargs: Dictionary of attributes to update
        :type kwargs: dict
        """
        for (key, value) in kwargs.items():
            self.__setattr__(key, value)

    def integrate(self, xs: npt.NDArray[float], vs: npt.NDArray[any], target: SequentialTargets) \
            -> tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Integrates using the Leapfrog Integrator.

        Parameters
        ----------
        :param xs: (N, d) array of positions
        :type xs: np.ndarray
        :param vs: (N, d) array of velocities
        :type vs: np.ndarray
        :param target: Target distribution
        :type target: SequentialTargets
        :return: Tuple of arrays (positions, velocities) of shape (N, T+1, d) and (N, T+1, d)
        :rtype: tuple
        """
        N = xs.shape[0]
        pos = np.zeros((N, self.T+1, self.x_dim))  # trajectories
        aux = np.zeros((N, self.T+1, self.x_dim))
        pos[:, 0] = xs
        aux[:, 0] = vs

        # first half momentum step
        vs = vs - 0.5 * self.step_size * self.dVdq(xs, target.param_new)

        for t in range(self.T-1):
            # full position step
            xs = xs + self.step_size * self.v_transform(vs)  # cho_solve((self.chol_mass, True), vs.T).T
            # full momentum step
            vs = vs - self.step_size * self.dVdq(xs, target.param_new)
            # store
            pos[:, t+1] = xs
            aux[:, t+1] = vs

        # last full position step
        xs = xs + self.step_size * self.v_transform(vs)  # cho_solve((self.chol_mass, True), vs.T).T
        # final half momentum step
        vs = vs - 0.5 * self.step_size * self.dVdq(xs, target.param_new)
        # store
        pos[:, self.T] = xs
        aux[:, self.T] = -vs  # flip the sign of the velocities
        return pos, aux

    def sample_auxiliaries(self, N: int, rng: Optional[np.random.Generator] = None) -> npt.NDArray[float]:
        """Samples auxiliary variables from the standard normal distribution.

        Parameters
        ----------
        :param N: Number of samples to draw
        :type N: int
        :param rng: Random number generator for reproducibility
        :type rng: np.random.Generator
        :return: Sampled auxiliary variables of shape (N, d)
        :rtype: np.ndarray
        """
        self.rng = rng if rng is not None else self.rng
        return self.chol_mass.dot(self.rng.normal(loc=0.0, scale=1.0, size=(self.v_dim, N))).T

    def eval_aux_logdens(self, vs: npt.NDArray[any]) -> npt.NDArray[float]:
        """Evaluates the log density of the auxiliary variables.

        Parameters
        ----------
        :param vs: Auxiliary variables at which we evaluate the log density, has shape (N, d)
        :type vs: np.ndarray
        :return: Log density of the auxiliary variables of shape (N, )
        :type: np.ndarray
        """
        return self.aux_nc - 0.5*np.einsum('ij,ji->i', vs, self.v_transform(vs).T)


class AMIntegrator(Integrator):

    def __init__(self, d: int, T: int, step_size: float, int_type: str = 'thug'):
        """Represents both THUG or SNUG integrator."""
        assert int_type in ['thug', 'snug'], "Type must be either 'thug' or 'snug'."
        super().__init__(x_dim=d, v_dim=d, T=T)
        self.step_size = step_size
        self.rng = np.random.default_rng(np.random.randint(low=1000, high=9999))
        self.int_type = int_type
        self.sign = 1.0 if self.int_type == 'thug' else -1.0

    def __repr__(self):
        name = "THUG" if self.int_type == 'thug' else "SNUG"
        return f"{name} Integrator with T={self.T} and step size={self.step_size}."

    def integrate(self, xs: npt.NDArray[float], vs: npt.NDArray[any], target: Filamentary):
        N = xs.shape[0]
        pos = np.zeros((N, self.T+1, self.x_dim))
        aux = np.zeros((N, self.T+1, self.v_dim))
        pos[:, 0] = xs
        aux[:, 0] = vs
        for t in range(self.T):
            xs = xs + 0.5*self.step_size*vs
            vs = self.sign*(vs - 2*project(xs, vs, target.manifold.jac))
            xs = xs + 0.5*self.step_size*vs
            pos[:, t+1] = xs
            aux[:, t+1] = vs
        return pos, aux

    def sample_auxiliaries(self, N: int, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else self.rng
        return rng.normal(loc=0.0, scale=1.0, size=(N, self.v_dim))

    def eval_aux_logdens(self, vs: npt.NDArray[any]):
        return -0.5*(np.linalg.norm(vs, axis=1)**2)


class IntegratorMixtureSameT:

    def __init__(self, *integrators: Integrator, mixture_probabilities: npt.NDArray[float]):
        assert len(integrators) == len(mixture_probabilities), "There must be one mixture weight for each integrator."
        assert np.sum(mixture_probabilities) == 1.0, "Mixture probabilities must sum to one."
        assert (mixture_probabilities >= 0).all(), "Mixture weights must all be non-negative"
        Ts = np.array([integrator.T for integrator in integrators])
        assert np.all(Ts == Ts[0]), "Every integrator must have the same number of integrator steps."
        self.T = Ts[0]
        self.integrators = {ix: integrator for ix, integrator in enumerate(integrators)}
        self.mixture_probabilities = mixture_probabilities
        self.rng = np.random.default_rng(np.random.randint(low=1000, high=9999))
        self.n_int = len(self.integrators)  # number of integrators
        self.pos_dim = self.integrators[0].x_dim
        self.aux_dim = self.integrators[0].v_dim

    def sample_iotas(self, N: int):
        return self.rng.choice(a=self.n_int, size=N, replace=True, p=self.mixture_probabilities)

    def integrate(self, xs: npt.NDArray[float], vs: npt.NDArray[float], iotas: npt.NDArray[float],
                  target: SequentialTargets):
        # sample to choose which particles will be integrated with which integrator
        N = xs.shape[0]
        # integrate the respective particles, create a total trajectory
        pos = np.zeros((N, self.T+1, self.pos_dim))
        aux = np.zeros((N, self.T+1, self.aux_dim))
        for i in range(self.n_int):
            p, a = self.integrators[i].integrate(xs[iotas == i], vs[iotas == i], target)
            pos[iotas == i] = p
            aux[iotas == i] = a
        return pos, aux

    def sample_auxiliaries(self, N: int, iotas: npt.NDArray[float], rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else self.rng
        aux = np.zeros((N, self.aux_dim))
        for i in range(self.n_int):
            aux[iotas == i] = self.integrators[i].sample_auxiliaries(N=np.sum(iotas == i), rng=rng)
        return aux

    def eval_aux_logdens(self, vs: npt.NDArray[any], iotas: npt.NDArray[float]):
        vals = np.zeros(len(vs))
        for i in range(self.n_int):
            vals[iotas == i] = self.integrators[i].eval_aux_logdens(vs=vs[iotas == i])
        return vals
