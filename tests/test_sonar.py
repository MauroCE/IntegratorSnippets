import numpy as np
from scipy.stats import multivariate_normal as mvn
from integrator_snippets.integrators import LeapfrogIntegrator
from integrator_snippets.distributions import Tempered
from integrator_snippets.monitoring import MonitorSingleIntSnippet
from integrator_snippets.adaptation import DummyAdaptation
from integrator_snippets.samplers import SingleIntegratorSnippet


def get_sonar_data():
    raw_data = np.loadtxt("../data/sonar.all-data",
                          delimiter=",", converters={60: lambda x: 1 if x == b"R" else 0})
    response = raw_data[:, -1]
    # Preprocess predictors
    predictors = np.atleast_2d(raw_data[:, :-1])
    rescaled_predictors = 0.5 * (predictors - np.mean(predictors, axis=0)) / np.std(predictors, axis=0)
    n, n_predictors = predictors.shape
    output = np.empty((n, n_predictors + 1))
    output[:, 0] = 1.0  # intercept
    output[:, 1:] = rescaled_predictors
    predictors = output
    return predictors, response


def log_prior(x_matrix):
    """The prior is a product of independent normals. The first component has scale 20 and the others 5."""
    assert len(x_matrix.shape) == 2, "Must be a matrix."
    return mvn(mean=np.zeros(61), cov=np.diag([20.0] + [5.0]*60)).logpdf(x_matrix)


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def log_sigmoid(x):
    """Log of the sigmoid function."""
    return -np.log(1 + np.exp(-x))


def grad_log_sigmoid(x):
    """Gradient of the log sigmoid function."""
    return np.exp(-x) / (1 + np.exp(-x))


if __name__ == "__main__":

    # Grab Sonar Data
    Z, y = get_sonar_data()  # (n_d, p), (n_d)  where n_d=208 and p=61

    # Settings
    p = Z.shape[1]  # 61
    N = 5000
    T = 20
    hmc_step_size = 0.1

    # Define the gradient of the potential
    def grad_neg_log_dens(X, gamma):
        """Expects X (N, p). For each x in X, do dot product with all Z. Output must also be (N, p)."""
        grad = np.zeros((N, p))
        yZ = y[:, None] * Z  # (n_d, p)
        for i, x in enumerate(X):
            yZx = (yZ @ x)[:, None]  # (n_d, 1)
            grad[i] = np.sum(grad_log_sigmoid(-yZx) * yZ, axis=0)  # (p, )
        return gamma * grad

    def log_likelihood(X):
        """Computes log likelihood for particles X of shape  (N, p)."""
        return np.sum(log_sigmoid(- y[None, :] * (X @ Z.T)), axis=1)  # (N, )

    # Integrator
    leapfrog = LeapfrogIntegrator(d=p, T=T, step_size=hmc_step_size, dVdq=grad_neg_log_dens)

    # Targets
    targets = Tempered(gamma=0.01, data={'Z': Z, 'y': y}, alpha=0.9, tol=1e-8)
    targets.log_prior = log_prior
    targets.log_likelihood = log_likelihood
    targets.log_dens_aux = leapfrog.eval_aux_logdens
    targets.sample_initial_particles = lambda n: np.random.randn(n, p) * np.array([20.0] + [5.0]*60)[None, :]

    # Monitors
    monitor = MonitorSingleIntSnippet(terminal_metric=1e-2, metric='pm')

    # Adaptators
    adaptator = DummyAdaptation()

    # Integrator Snippet
    int_snip = SingleIntegratorSnippet(
        N=N, integrator=leapfrog, targets=targets, monitor=monitor, adaptator=adaptator, verbose=True, plot_every=100
    )
    out = int_snip.sample()
