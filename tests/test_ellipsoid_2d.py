import numpy as np
from integrator_snippets.manifolds import Ellipsoid
from integrator_snippets.distributions import Filamentary, FilamentaryFixedTolSeq
from integrator_snippets.integrators import AMIntegrator, IntegratorMixtureSameT, LeapfrogIntegrator
from integrator_snippets.monitoring import MonitorMixtureIntSnippet, MonitorSingleIntSnippet
from integrator_snippets.adaptation import DummyAdaptation, MixtureStepSizeAdaptorSA, SingleStepSizeAdaptorSA
from integrator_snippets.samplers import MixtureIntegratorSnippetSameT, SingleIntegratorSnippet, GHUMS
from integrator_snippets.mixture_weights import UniformMixtureWeights
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Settings
    d = 50                              # Dimension of the ambient space
    center = np.zeros(d)                # Center of the ellipsoid
    Sigma = np.diag([1., .1]*(d//2))    # Covariance matrix of the ellipsoid
    z = np.exp(-30.0)                   # Specifies which level set of MVN the ellipsoid corresponds to

    # Generate points on ellipsoid
    ellipsoid = Ellipsoid(mu=np.zeros(d), cov=Sigma, z=z)
    points = ellipsoid.sample(1000)
    print("Maximum distance of sampled points from ellipsoid: ", np.max(abs(ellipsoid.f(points))))

    # Filamentary distribution = standard normal times a uniform kernel
    eps0 = 350
    epsP = 1e-5
    epsilons = np.geomspace(start=eps0, stop=epsP, num=300)
    filamentary = FilamentaryFixedTolSeq(manifold=ellipsoid, epsilons=epsilons, kernel='uniform')

    # initial particles
    seed = 748932
    rng = np.random.default_rng(seed=seed)
    filamentary.sample_initial_particles = lambda n_particles: rng.normal(loc=0.0, scale=1.0, size=(n_particles, d))
    filamentary.base_log_dens_x = lambda x: -0.5*(np.linalg.norm(x, axis=-1)**2)
    filamentary.log_dens_aux = lambda v: -0.5*(np.linalg.norm(v, axis=-1)**2)

    # GHUMS settings
    N = 5000
    T = 30

    thug_step = 0.1
    snug_step = 0.01
    p_thug = 0.8
    # Integrator
    thug = AMIntegrator(d=filamentary.manifold.d, T=T, step_size=thug_step, int_type='thug')
    snug = AMIntegrator(d=filamentary.manifold.d, T=T, step_size=snug_step, int_type='snug')
    integrators = IntegratorMixtureSameT(thug, snug, mixture_probabilities=np.array([p_thug, 1-p_thug]))
    # Monitors
    thug_monitor = MonitorSingleIntSnippet(terminal_metric=1e-2, metric='mip')
    snug_monitor = MonitorSingleIntSnippet(terminal_metric=1e-2, metric='mip')
    monitors = MonitorMixtureIntSnippet(thug_monitor, snug_monitor)
    # Adaptation
    thug_adaptor = DummyAdaptation()
    snug_adaptor = DummyAdaptation()
    adaptators = MixtureStepSizeAdaptorSA(thug_adaptor, snug_adaptor)
    mix_weights = UniformMixtureWeights(T=T)
    ghums = MixtureIntegratorSnippetSameT(
        N=N, int_mixture=integrators, targets=filamentary, monitors=monitors, adaptators=adaptators,
        mixture_weights=mix_weights, max_iter=5000, verbose=True, plot_every=100, rng=rng)
    ghums.sample()

    # Plot ESS
    fig, ax = plt.subplots()
    ess_vals = ghums.monitors.rel_ess_values
    ax.plot(epsilons[:len(ess_vals)], ess_vals)
    ax.set_xscale('log')
    plt.show()

    # Plot AP
    fig, ax = plt.subplots()
    pm_thug = ghums.monitors.pms['0']
    ax.plot(epsilons[:len(pm_thug)], pm_thug)
    ax.set_xscale('log')
    plt.show()

    fig, ax = plt.subplots()
    pm_snug = ghums.monitors.pms['1']
    ax.plot(epsilons[:len(pm_snug)], pm_snug)
    ax.set_xscale('log')
    plt.show()



