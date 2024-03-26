import numpy as np
import numpy.typing as npt
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal as mvn


class Manifold:

    def __init__(self, d: int, m: int):
        """Implements an implicitly-defined manifold.

        Parameters
        ----------
        :param d: Dimension of the ambient space
        :type d: int
        :param m: Co-dimension of the manifold, or equivalently, number of constraints
        :type m: int

        Notes
        -----
        This class implements manifolds implicitly defined by a smooth function f:\\mathbb{R}^d \to \\mathbb{R}^m, where
         d > m >= 1.

        \\mathcal{M} = \\{x \\in \\mathbb{R}^d : f(x) = 0 \\}

        The space \\mathbb{R}^d is disintegrated into a union of level sets of f using the Co-Area formula. We assume
        that our manifold of interest corresponds to f^{-1}(0).
        """
        assert isinstance(d, int), "Dimensionality of the ambient space must be an integer."
        assert isinstance(m, int), "Co-dimension of the manifold must be an integer."
        assert d > m, "Ambient space dimension must be strictly larger than co-dimension."
        assert m >= 1, "Co-dimensionality of the manifold must be at least one."
        self.m = m  # number of constraints / co-dimension
        self.d = d  # number of variables / dimension of ambient space
        self.dim = self.d - self.m  # Manifold dimension
        if self.m == 1:
            self.project = self._univariate_project  # only one way to project when m = 1
            self.distances = self._univariate_distances_from_manifold
        else:
            self.project = self._qr_project
            self.distances = self._multivariate_distances_from_manifold

    def f(self, xs: npt.NDArray[float]) -> npt.NDArray[float]:
        """Vectorised constraint function. Computes constraint at each row of xs.

        Parameters
        ----------
        :param xs: Matrix whose rows are in the ambient space where we wish to evaluate the constraint function f
        :type xs: np.NDArray
        :return: Constraint function evaluated at each row of x, it's an array of shape (N, m)
        :rtype: np.NDArray
        """
        raise NotImplementedError("Constraint function not implemented.")

    def jac(self, xs: npt.NDArray[float]) -> npt.NDArray[float]:
        """Jacobian of the constraint evaluated at each row of xs. Returns a (N, m, n) tensor where N is the number of
        particles, m is the co-dimension of the manifold and n is the dimension of the ambient space. The number N is
        the length of the first dimension of the xs.

        Parameters
        ----------
        :param xs: Matrix whose rows are points in the ambient space at which we wish to compute the Jacobian
        :type xs: np.ndarray
        :return: Tensor where every slice in the first dimension is a Jacobian for the corresponding row of xs
        :rtype: np.ndarray
        """
        raise NotImplementedError("Jacobian of the function not implemented.")

    def _qr_project(self, vs: npt.NDArray[float], xs: npt.NDArray[float]) -> npt.NDArray[float]:
        """Same as `self._qr_project` but computes the projection for all particles at once. This should be used in an
        SMC sampler to vectorise computations.

        Parameters
        ----------
        :param vs: Velocities to be projected, should have shape (N, d)
        :type vs: np.ndarray
        :param xs: Points at which to project respective velocities, should have shape (N, d)
        :type xs: np.ndarray
        :return: Projected velocities stored in a matrix of shape (N, d)
        :rtype: np.ndarray
        """
        qs = np.linalg.qr(np.transpose(self.jac(xs), axes=(0, 2, 1)), mode='reduced')[0]
        return np.einsum(
            'ijk,ik->ij',
            qs,
            np.einsum('ijk,ik->ij', np.transpose(qs, axes=(0, 2, 1)), vs)
        )

    def _univariate_project(self, vs: npt.NDArray[float], xs: npt.NDArray[float]) -> npt.NDArray[float]:
        """Vectorised version of the univariate projection function.

        Parameters
        ----------
        :param vs: Velocities to be projected, should have shape (N, d)
        :type vs: np.ndarray
        :param xs: Points at which to project respective velocities, should have shape (N, d)
        :type xs: np.ndarray
        :return: Projected velocities stored in a matrix of shape (N, d)
        :rtype: np.ndarray
        """
        gs = self.jac(xs).squeeze(axis=1)
        gs_hat = gs / np.linalg.norm(gs, axis=1, keepdims=True)  # Normalize each row (gradient)
        return np.multiply(gs_hat, np.einsum('ij,ij->i', vs, gs_hat)[:, None])

    def _univariate_distances_from_manifold(self, xs: npt.NDArray[float]) -> npt.NDArray[float]:
        """Computes the maximum distance from the manifold of all particles in xs. This version is for m = 1.

        Parameters
        ----------
        :param xs: Vector where each element is a point in the ambient space.
        :type xs: np.ndarray
        :return: Distances of points from the manifold
        :rtype: np.ndarray
        """
        return np.abs(self.f(xs))

    def _multivariate_distances_from_manifold(self, xs: npt.NDArray[float]) -> npt.NDArray[float]:
        """Same as `_univariate_distance_from_manifold` but to use when m > 1.

        Parameters
        ----------
        :param xs: Matrix where each row is a point in the ambient space.
        :type xs: np.ndarray
        :return: Distances from the manifold
        :rtype: np.ndarray
        """
        return np.linalg.norm(self.f(xs), axis=1)

    def __repr__(self):
        return "Abstract Manifold Class."


class Ellipsoid(Manifold):

    def __init__(self, mu: npt.NDArray[float], cov: npt.NDArray[float], z: float):
        """Ellipsoid of dimension n-1 in n-dimensional ambient space. We view the Ellipsoid as a contour of an n-dim
        normal distribution with parameters mu and cov.

        Parameters
        ----------
        :param mu: Mean of the multivariate normal distribution, corresponds to the center of the ellipsoid
        :type mu: np.ndarray
        :param cov: Covariance matrix of multivariate normal, defines the covariance structure of the ellipsoid
        :type cov: np.ndarray
        :param z: Level-set value of the multivariate normal density
        :type z: float
        """
        super().__init__(d=len(mu), m=1)
        self.mu = mu
        self.cov = cov
        self.z = z
        self.mvn = mvn(mean=self.mu, cov=self.cov)

    def f(self, xs: npt.NDArray[float]) -> npt.NDArray[float]:
        """Constraint function for ellipsoid. This is vectorised and expects a matrix input of shape (N, d).

        Parameters
        ----------
        :param xs: Matrix whose rows are in the ambient space where we wish to evaluate the constraint function f
        :type xs: np.NDArray
        :return: Constraint function evaluated at each row of x, it's an array of shape (N, m)
        :rtype: np.NDArray
        """
        assert len(xs.shape) == 2, "Must be a matrix."
        return self.mvn.logpdf(xs).reshape(xs.shape[0], self.m) - np.log(self.z)

    def jac(self, xs) -> npt.NDArray[float]:
        """Vectorised Jacobian. Works on an input of shape (N, d).

        Parameters
        ----------
        :param xs: Matrix whose rows are points in the ambient space at which we wish to compute the Jacobian
        :type xs: np.ndarray
        :return: Tensor where every slice in the first dimension is a Jacobian for the corresponding row of xs
        :rtype: np.ndarray
        """
        return - np.linalg.solve(self.cov, (xs - self.mu).T).T[:, np.newaxis, :]  # insert extra axis of size self.m=1

    def sample(self, N: int):
        """Samples N points from the ellipsoid. Can be used for debugging.

        Parameters
        ----------
        :param N: Number of points to sample
        :type N: int
        :return: Matrix of samples, shape (N, d)
        :rtype: np.ndarray
        """
        # find them by optimization
        samples = np.zeros((N, self.d))
        i = 0
        while i < N:
            # sample a random point
            x = self.mvn.rvs()
            # use optimization to find a point such that f(x) = 0
            root, _, ier, _ = fsolve(
                func=lambda xx: [self.mvn.logpdf(xx) - np.log(self.z), 0.0],
                x0=x,
                full_output=True)
            if ier == 1:
                samples[i] = root
                i += 1
        return samples

    def __repr__(self):
        return "{}-dim Ellipsoid".format(self.dim)


class GK(Manifold):

    def __init__(self, y_star: npt.NDArray[float]):
        """Manifold for the G-and-K distribution.

        Parameters
        ----------
        :param y_star: Observed data, has shape (m, )
        :type y_star: np.ndarray

        """
        super().__init__(d=4+len(y_star), m=len(y_star))
        self.y_star = y_star

    def f(self, xs: npt.NDArray[float]) -> npt.NDArray[float]:
        """Computes the (vectorised) constraint function for the G-and-K manifold.

        Parameters
        ----------
        :param xs: Matrix whose rows we wish to evaluate the constraint at
        :type xs: npt.NDArray[float]
        :return: Constraint function evaluated at each row of xs
        :rtype: npt.NDArray[float]
        """
        # Parameters are (N, 1) dimensional
        a = xs[:, :0]
        b = xs[:, 1:2]
        g = xs[:, 2:3]
        k = xs[:, 3:4]
        # Latents are (N, m) dimensional
        z = xs[:, 4:]
        return a + b*(1 + 0.8*np.tanh(g*z/2))*z*(1 + z**2)**k - self.y_star[None, :]  # (N, m)

    def jac(self, xs: npt.NDArray[float]) -> npt.NDArray[float]:
        """Jacobian of the constraint function.

        Parameters
        ----------
        :param xs: Matrix whose rows are points in the ambient space at which we wish to compute the Jacobian
        :type xs: npt.NDArray[float]
        :return: Jacobian of the constraint function evaluated at each row of xs
        :rtype: npt.NDArray[float]
        """
        pass

    def __repr__(self):
        return """GK Manifold with {} datapoints.""".format(self.m)
