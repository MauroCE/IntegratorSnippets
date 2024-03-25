import numpy as np
from manifolds import Ellipsoid
from distributions import Filamentary
from integrators import AMIntegrator, IntegratorMixtureSameT, LeapfrogIntegrator
from monitoring import MonitorMixtureIntSnippet, MonitorSingleIntSnippet
from adaptation import DummyAdaptation, MixtureStepSizeAdaptorSA, SingleStepSizeAdaptorSA
from integrator_snippets import MixtureIntegratorSnippetSameT, SingleIntegratorSnippet
from utils import grad_neg_normal_log_kernel
import numpy.typing as npt


if __name__ == "__main__":
    # Settings
    d = 2
    N = 1000
    T = 20
    thug_step_size = 0.1
    snug_step_size = 0.1

    # Manifold
    mu = np.zeros(2)
    Sigma = np.array([[1.0, 0.0], [0.0, 0.1]])
    c = 0.05
    manifold = Ellipsoid(mu=mu, cov=Sigma, z=c)

    # Integrator
    thug = AMIntegrator(d=d, T=T, step_size=thug_step_size, int_type='thug')
    snug = AMIntegrator(d=d, T=T, step_size=snug_step_size, int_type='snug')
    integrators = IntegratorMixtureSameT(thug, snug, mixture_probabilities=np.array([0.8, 0.2]))

    # Target
    targets = Filamentary(manifold=manifold, eps=1000, kernel='uniform', coeff=1.0)
    targets.base_log_dens_x = lambda x: -0.5*(np.linalg.norm(x, axis=-1)**2)
    targets.sample_initial_particles = lambda n_particles: np.random.randn(n_particles, d)
    targets.log_dens_aux = thug.eval_aux_logdens

    # Monitors
    thug_monitor = MonitorSingleIntSnippet(terminal_metric=1e-2, metric='pm')
    snug_monitor = MonitorSingleIntSnippet(terminal_metric=1e-2, metric='pm')
    monitors = MonitorMixtureIntSnippet(thug_monitor, snug_monitor)

    # Adaptors
    thug_adaptator = DummyAdaptation()
    snug_adaptator = SingleStepSizeAdaptorSA(target_metric_value=0.5, metric='mip',
                                             max_step=10., min_step=0.000001, lr=0.5)
    adaptators = MixtureStepSizeAdaptorSA(thug_adaptator, snug_adaptator)

    # Integrator Snippet
    ghums = MixtureIntegratorSnippetSameT(N=N, int_mixture=integrators, targets=targets, monitors=monitors,
                                          adaptators=adaptators, max_iter=5000, verbose=True, plot_every=100)
    out = ghums.sample()

    # Try the same but with HMC
    # def neg_log_base_dens(x_matrix) -> npt.NDArray[float]:
    #     """Base negative log density for N inputs."""
    #     assert len(x_matrix.shape) == 2, "must be two dimensional input"
    #     return 0.5 * (np.linalg.norm(x_matrix, axis=-1) ** 2)
    # def grad_neg_log_base_dens(x_matrix):
    #     """Gradient"""
    #     return x_matrix
    # def dVdq(x_matrix, eps):
    #     return grad_neg_normal_log_kernel(x_matrix, eps, jac=manifold.jac, f=manifold.f) + grad_neg_log_base_dens(x_matrix)
    # leapfrog = LeapfrogIntegrator(d=d, T=T, step_size=0.1, dVdq=dVdq)
    # targets = Filamentary(manifold=manifold, eps=1000, kernel='normal', coeff=0.9)
    # targets.base_log_dens_x = lambda x: -0.5*(np.linalg.norm(x, axis=-1)**2)
    # targets.sample_initial_particles = lambda n_particles: np.random.randn(n_particles, d)
    # targets.log_dens_aux = thug.eval_aux_logdens
    # monitor = MonitorSingleIntSnippet(terminal_metric=1e-2, metric='pm')
    # adaptator = DummyAdaptation()
    # int_snip = SingleIntegratorSnippet(
    #     N=N,
    #     integrator=leapfrog,
    #     targets=targets,
    #     monitor=monitor,
    #     adaptator=adaptator,
    #     max_iter=1000,
    #     verbose=True,
    #     plot_every=2
    # )
    # out_hmc = int_snip.sample()
