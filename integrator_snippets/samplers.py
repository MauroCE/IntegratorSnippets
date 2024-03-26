import numpy as np
from matplotlib import rc
from typing import Optional
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from .utils import setup_rng
from .distributions import SequentialTargets
from .integrators import Integrator, IntegratorMixtureSameT
from .monitoring import Monitor, MonitorSingleIntSnippet, MonitorMixtureIntSnippet
from .adaptation import AdaptationStrategy, SingleStepSizeAdaptorSA, MixtureStepSizeAdaptorSA
from .mixture_weights import MixtureWeights, UniformMixtureWeights


class AbstractIntegratorSnippet:

    def __init__(self):
        """Abstract class for an integrator snippet. Provides a common base class for all integrator snippets."""
        self.pos = None
        self.aux = None
        self.n = 1

    def setup_sampling(self):
        raise NotImplementedError("Method to setup sampling has not been implemented yet.")

    def setup_storage(self):
        raise NotImplementedError("Method to setup storage has not been implemented yet.")

    def termination_criterion(self):
        raise NotImplementedError("Termination criterion not implemented yet.")

    def output(self):
        raise NotImplementedError("Output not implemented yet.")

    def select_next_target(self):
        raise NotImplementedError("Method to select next tolerance not implemented yet.")

    def construct_trajectories(self):
        raise NotImplementedError("Method to construct trajectories not implemented yet.")

    def compute_weights(self):
        raise NotImplementedError("Method to compute weights not implemented yet.")

    def resample(self):
        raise NotImplementedError("Method to resample not implemented yet.")

    def compute_metrics(self):
        raise NotImplementedError("Method to compute acceptance metrics not implemented yet.")

    def adapt_parameters(self):
        raise NotImplementedError("Method to adapt parameters not implemented yet.")

    def refresh_auxiliaries(self):
        raise NotImplementedError("Method to refresh auxiliary variables not implemented yet.")

    def plot_particles(self):
        pass

    def sample(self):
        """Samples using an integrator snippet. All extensions discussed in the paper, follow this same structure."""
        self.setup_sampling()
        self.setup_storage()

        try:
            while not self.termination_criterion():
                self.select_next_target()
                self.construct_trajectories()
                self.compute_weights()
                self.resample()
                self.compute_metrics()
                self.adapt_parameters()
                self.refresh_auxiliaries()
                self.plot_particles()
                self.n += 1

        except (KeyboardInterrupt, OverflowError):
            print("Error occurred, exiting.")
            return self.output()

        return self.output()


class SingleIntegratorSnippet(AbstractIntegratorSnippet):

    def __init__(self, N: int, integrator: Integrator, targets: SequentialTargets,
                 monitor: Monitor = MonitorSingleIntSnippet(terminal_metric=1e-2, metric='pm'),
                 adaptator: AdaptationStrategy = SingleStepSizeAdaptorSA(target_metric_value=0.3, metric='mip'),
                 mixture_weights: MixtureWeights = UniformMixtureWeights(T=2),
                 max_iter: int = 1000,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 plot_every: int = 5):
        """Initialises the integrator snippet. This class should be used for integrator snippets using a single
        integrator.

        Parameters
        ----------
        :param N: Number of particles
        :type N: int
        :param integrator: Integrator used to construct trajectories
        :type integrator: Integrator
        :param targets: Sequential targets targeted by the integrator snippet
        :type targets: SequentialTargets
        :param monitor: Computes and stores metrics to assess performance and termination of the integrator snippet
        :type monitor: Monitor
        :param adaptator: Adapts the parameters of the integrator based on the metrics computed by the monitor
        :type adaptator: AdaptationStrategy
        :param max_iter: Maximum number of iterations
        :type max_iter: int
        :param seed: Random seed for reproducibility
        :type seed: int
        :param verbose: Whether to print information during the sampling, can be used for debugging or monitoring
        :type verbose: bool
        :param plot_every: How often to plot the particles, can be used for debugging
        :type plot_every: int
        """
        super().__init__()
        self.N = N
        self.T = integrator.T
        self.integrator = integrator
        self.targets = targets
        self.pos = None  # position variables. Variables of the real, marginal target (N, d)
        self.aux = None  # auxiliary variables, used to aid integration (N, d)
        self.pos_nk = None  # (N, T+1, d)
        self.aux_nk = None  # (N, T+1, d)
        self.logw = np.full((N, self.T+1), - np.log(N*(self.T+1)))  # log un-normalised weights, init as uniform
        self.W = np.exp(self.logw - logsumexp(self.logw))  # normalised weights
        self.logw_folded = - np.log(self.T+1) + logsumexp(self.logw, axis=1)
        self.indices = None
        self.trajectory_indices = None
        self.particle_indices = None
        self.rng = setup_rng(seed=seed, rng=None)
        self.max_iter = max_iter
        self.monitor = monitor
        self.adaptator = adaptator
        self.verbose = verbose
        self.print = print if self.verbose else lambda *a, **k: None
        self.plot_every = plot_every
        # Update mixture weights T, to allow for a sensible initialization
        self.mixture_weights = mixture_weights
        self.mixture_weights.T = self.T

    def setup_sampling(self):
        """Samples initial particles. For more complex variants, this should be overridden."""
        self.pos = self.targets.sample_initial_particles(self.N)
        self.aux = self.integrator.sample_auxiliaries(N=self.N, rng=self.rng)
        self.print("Particles initialised.")

    def resample(self):
        """Resamples N particles out of N(T+1)."""
        self.indices = self.rng.choice(a=self.N*(self.T + 1), size=self.N, replace=True, p=self.W.ravel())
        self.particle_indices, self.trajectory_indices = np.unravel_index(
            self.indices, (self.N, self.T + 1)
        )
        self.pos = self.pos_nk[self.particle_indices, self.trajectory_indices]
        self.print("\tParticles resampled.")

    def compute_weights(self):
        """Computes log weights for each of the N(T+1) particles."""
        self.logw = self.targets.logw(self.pos_nk, self.aux_nk) + self.mixture_weights.log_weights()
        self.W = np.exp(self.logw - logsumexp(self.logw))
        self.logw_folded = - np.log(self.T+1) + logsumexp(self.logw, axis=1)
        self.print("\tWeights computed. Finite: ", np.isfinite(self.W).sum())

    def construct_trajectories(self):
        """Constructs trajectories from each of the N particles by iterating the integrator T times."""
        self.pos_nk, self.aux_nk = self.integrator.integrate(xs=self.pos, vs=self.aux, target=self.targets)
        self.print("\tTrajectories constructed.")

    def refresh_auxiliaries(self):
        """Refreshes any auxiliary variables used by the integrator."""
        self.aux = self.integrator.sample_auxiliaries(N=self.N, rng=self.rng)
        self.print("\tAuxiliaries refreshed.")

    def select_next_target(self):
        """Select the target for the next iteration, either based on tempering, a fixed sequence or ABC-style."""
        self.targets.update_parameter(self.__dict__)
        self.print("\tNext target selected: ", self.targets.param_new)

    def termination_criterion(self):
        """Checks if the termination conditions are met. Target distributions check if we have reached the final target
        distribution, whereas the monitor checks if we have reached a terminal metric."""
        self.print("Iteration: ", self.n)
        return (self.n > self.max_iter) or self.targets.terminate() or self.monitor.terminate()

    def compute_metrics(self):
        """Computes metrics based on Monitor. This allows a lot of flexibility from the user or for debugging."""
        self.monitor.update_metrics(self.__dict__)
        self.print("\tMetrics computed.")

    def adapt_parameters(self):
        """Adapts (hyper)parameters of the integrators based on the metrics computed by the monitor."""
        adaptation_dict = self.adaptator.adapt(self.__dict__)
        for (attribute_key, adapted_value) in adaptation_dict.items():
            self.integrator.__dict__[attribute_key] = adapted_value
            self.print("\t" + attribute_key, " adapted to ", adapted_value)
        # Update mixture weights' T
        self.T = self.integrator.T
        self.mixture_weights.T = self.T

    def setup_storage(self):
        """Should be overridden if we want to store additional variables."""
        pass

    def plot_particles(self):
        """Plots particles at `plot_every` iterations for debugging."""
        if self.n % self.plot_every == 0:
            rc('font', **{'family': 'STIXGeneral'})
            with plt.style.context("dark_background"):
                fig, ax = plt.subplots()
                ax.scatter(*self.pos.T, label="particles", s=2)
                ax.grid(True, color='gainsboro')
                ax.set_ylim([-4, 4])
                ax.set_xlim([-10, 10])
                ax.set_xlabel(r"$\mathregular{x_0}$", fontsize=20)
                ax.set_ylabel(r"$\mathregular{x_1}$", fontsize=20)
                ax.set_aspect('equal')
                ax.legend()
                ax.set_title("n = {}".format(self.n))
                plt.show()

    def output(self):
        """Defines the output of the integrator snippet."""
        return {
            'monitor': self.monitor,
            'pos': self.pos,
            'aux': self.aux,
            'logw': self.logw,
            'W': self.W,
            'n': self.n
        }


class MixtureIntegratorSnippetSameT(AbstractIntegratorSnippet):

    def __init__(self, N: int, int_mixture: IntegratorMixtureSameT, targets: SequentialTargets,
                 monitors: MonitorMixtureIntSnippet,
                 adaptators: MixtureStepSizeAdaptorSA,
                 mixture_weights: MixtureWeights = UniformMixtureWeights(T=2),
                 max_iter: int = 1000,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 plot_every: int = 5):
        """
        Initialises the integrator snippet using a mixture of integrators, where each integrator has the same number of
        integration steps.

        :param N: Number of particles
        :type N: int
        :param int_mixture: Integrators used to construct trajectories
        :type int_mixture: IntegratorMixtureSameT
        :param targets: Generator for the sequence of targets
        :type targets: SequentialTargets
        :param monitors: Computes and stores metrics to assess performance and termination of the integrator snippet
        :type monitors: MonitorMixtureIntSnippet
        :param adaptators: Adapts the parameters of the integrator based on the metrics computed by the monitor
        :type adaptators: MixtureStepSizeAdaptorSA
        :param max_iter: Maximum number of iterations
        :type max_iter: int
        :param seed: Random seed for reproducibility
        :type seed: int
        :param verbose: Whether to print information during the sampling, can be used for debugging or monitoring
        :type verbose: bool
        :param plot_every: How often to plot the particles, can be used for debugging
        :type plot_every: int
        """
        super().__init__()
        self.N = N
        self.T = int_mixture.T
        self.integrators = int_mixture
        self.targets = targets
        self.iotas = None  # (N, ) identify which particle corresponds to which integrator
        self.pos = None  # position variables. Variables of the real, marginal target (N, d)
        self.aux = None  # auxiliary variables, used to aid integration (N, d)
        self.pos_nk = None  # (N, T+1, d)
        self.aux_nk = None  # (N, T+1, d)
        self.logw = None  # log un-normalised weights
        self.W = None  # normalised weights
        self.indices = None
        self.trajectory_indices = None
        self.particle_indices = None
        self.rng = setup_rng(seed=seed, rng=None)
        self.max_iter = max_iter
        self.monitors = monitors
        self.adaptators = adaptators
        self.verbose = verbose
        self.print = print if self.verbose else lambda *a, **k: None
        self.plot_every = plot_every
        # setup mixture weights
        self.mixture_weights = mixture_weights
        self.mixture_weights.T = self.T

    def setup_sampling(self):
        """Samples initial particles."""
        self.pos = self.targets.sample_initial_particles(self.N)
        self.iotas = self.integrators.sample_iotas(self.N)
        self.aux = self.integrators.sample_auxiliaries(N=self.N, iotas=self.iotas, rng=self.rng)
        self.print("Particles initialised.")

    def setup_storage(self):
        """Should be overridden if we want to store additional variables."""
        pass

    def termination_criterion(self):
        """Checks if the termination conditions are met, for each integrator and current target."""
        result = (self.n > self.max_iter) or self.targets.terminate() or self.monitors.terminate()
        if not result:
            self.print("Iteration: ", self.n)
        else:
            if self.n > self.max_iter:
                self.print("Maximum iteration reached.")
            elif self.targets.terminate():
                self.print("Target distribution reached.")
            else:
                indices = np.where([monitor.terminate() for monitor in self.monitors.monitors])[0]
                for ix in indices:
                    self.print("Terminal metric reached for", self.integrators.integrators[ix])
            print("-" * 50)
        return result

    def select_next_target(self):
        """Select the next target using the distribution."""
        self.targets.update_parameter(self.__dict__)
        self.print("\tNext target selected: ", self.targets.param_new)

    def construct_trajectories(self):
        """Constructs trajectories using the appropriate integrators. We can do things easily because here we assume
        all integrators use the same number of integration steps."""
        self.pos_nk, self.aux_nk = self.integrators.integrate(xs=self.pos, vs=self.aux, iotas=self.iotas,
                                                              target=self.targets)
        self.print("\tTrajectories constructed.")

    def compute_weights(self):
        """Computes log weights."""
        self.logw = self.targets.logw(self.pos_nk, self.aux_nk) + self.mixture_weights.log_weights()
        self.W = np.exp(self.logw - logsumexp(self.logw))
        self.print("\tWeights computed.")

    def resample(self):
        """Resamples N particles out of N(T+1)."""
        self.indices = self.rng.choice(a=self.N*(self.T+1), size=self.N, replace=True, p=self.W.ravel())
        self.particle_indices, self.trajectory_indices = np.unravel_index(
            self.indices, (self.N, self.T+1)
        )
        self.pos = self.pos_nk[self.particle_indices, self.trajectory_indices]
        self.print("\tParticles resampled.")

    def compute_metrics(self):
        """Computes metrics based on Monitors."""
        # Each Monitor "monitors" a different integrator. The variables iotas determine which particle uses which
        # integrator. Therefore, different monitors require knowing which particles are using which integrator to
        # compute the correct metrics.
        self.monitors.update_metrics(self.__dict__)
        self.print("\tMetrics computed.")

    def adapt_parameters(self):
        """Adapts parameters of each integrator."""
        adaptation_dicts = self.adaptators.adapt(self.__dict__)
        for ix, adapt_dict in adaptation_dicts.items():
            for (key, adapted_value) in adapt_dict.items():
                self.integrators.integrators[ix].__dict__[key] = adapted_value
                self.print("\t" + key, " adapted to ", "{:.5f}".format(adapted_value),
                           " for integrator ", self.integrators.integrators[ix])

    def refresh_auxiliaries(self):
        """Refreshes auxiliaries of each integrator."""
        self.aux = self.integrators.sample_auxiliaries(N=self.N, iotas=self.iotas, rng=self.rng)
        self.iotas = self.integrators.sample_iotas(N=self.N)
        self.print("\tAuxiliaries and iotas refreshed.")

    def plot_particles(self):
        """Plots particles at `plot_every` iterations for debugging."""
        if self.n % self.plot_every == 0:
            rc('font', **{'family': 'STIXGeneral'})
            with plt.style.context("dark_background"):
                fig, ax = plt.subplots()
                # plot some points showing the ellipse
                ax.scatter(*self.pos.T, label="particles", s=5, c='lightblue', marker='o', ec='navy')
                ax.grid(True, color='gainsboro')
                ax.set_ylim([-3, 3])
                ax.set_xlim([-3, 3])
                ax.set_xlabel(r"$\mathregular{x_0}$", fontsize=20)
                ax.set_ylabel(r"$\mathregular{x_1}$", fontsize=20)
                ax.set_aspect('equal')
                ax.legend()
                ax.set_title("n = {}".format(self.n))
                plt.show()

    def output(self):
        """Defines the output of the integrator snippet."""
        return {
            'monitor': self.monitors,
            'pos': self.pos,
            'aux': self.aux,
            'logw': self.logw,
            'W': self.W,
            'n': self.n
        }


class MixtureIntegratorSnippet(MixtureIntegratorSnippetSameT):

    def __init__(self, N: int, int_mixture: IntegratorMixtureSameT, targets: SequentialTargets,
                 monitors: MonitorMixtureIntSnippet,
                 adaptators: MixtureStepSizeAdaptorSA,
                 mixture_weights: MixtureWeights = UniformMixtureWeights(T=2),
                 max_iter: int = 1000,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 plot_every: int = 5):
        """Implements a mixture of integrators, who can have different number of integration steps."""
        super().__init__()
        pass
