import numpy as np
from scipy.special import logsumexp


class Monitor:

    def __init__(self):
        """This class should do two things:
        1. Compute metrics for integrator snippets.
        2. Check if we reached termination based on those metrics.
        """
        self.proportion_moved = 1.0
        self.proportion_resampled = 1.0
        self.particle_diversity = 1.0
        self.median_index_proportion = 1.0
        self.median_path_diversity = 1.0

    def update_metrics(self, attributes: dict):
        """Updates the metrics based on the attributes of the monitor, which will be in the attributes of the
         integrator snippet passed.

         Parameters
         ----------
         :param attributes: Dictionary with the attributes of the integrator snippet
         :type attributes: dict
         """
        raise NotImplementedError

    def terminate(self) -> bool:
        """Returns True if termination is reached, based on the metrics computed.

        Parameters
        ----------
        :return: Whether termination is reached
        :rtype: bool
        """
        raise NotImplementedError


class MonitorSingleIntSnippet(Monitor):

    def __init__(self, terminal_metric: float = 1e-2, metric: str = 'pm'):
        """Monitors progress and computes metrics for an integrator snippets using a single integrator.

        Parameters
        ----------
        :param terminal_metric: Metric value that determines termination of the integrator snippet
        :type terminal_metric: float
        :param metric: Metric to use for termination. Must be one of 'pm', 'mip', 'mpd' or None.
        :type metric: str
        """
        super().__init__()
        assert metric in {'pm', 'mip', 'mpd', None}, "Metric must be one of 'pm', 'mip', 'mpd' or None."
        self.terminal_metric = terminal_metric  # could be pm, mid, mpd
        self.metric = metric
        # choose the metric to test during termination based on self.metric
        match self.metric:
            case "pm":
                self.grab_metric = lambda: self.proportion_moved
            case "mip":
                self.grab_metric = lambda: self.median_index_proportion
            case "mpd":
                self.grab_metric = lambda: self.median_path_diversity
            case None:
                self.grab_metric = lambda: np.inf
        self.esjd_mean = None  # expected squared jump distance for the mean function
        self.ess_mubar = None

    def update_metrics(self, attributes: dict):
        """Updates and stores several metrics based on the attributes of the integrator snippets.

        Parameters
        ----------
        :param attributes: Dictionary with the attributes of the integrator snippet
        :type attributes: dict

        Notes
        -----
        Currently updates the following metrics:
        1. Proportion of Particles Moved
        2. Proportion of Particles resampled
        3. Particle Diversity
        4. Median Index Proportion
        5. Median Path Diversity
        6. ESS for mu bar
        """
        N = attributes['N']
        T = attributes['T']
        self.proportion_moved = np.sum(attributes['trajectory_indices'] >= 1) / N
        self.proportion_resampled = len(np.unique(attributes['indices'])) / N
        self.particle_diversity = (len(np.unique(attributes['particle_indices'])) - 1) / (N - 1)
        self.median_index_proportion = np.median(attributes['trajectory_indices']) / T
        self.median_path_diversity = np.sqrt(self.particle_diversity * self.median_index_proportion)
        # Compute ESS for mu bar. This requires obtaining the folded weights from the unfolded ones
        logw_folded = logsumexp(attributes['logw'] + attributes['mixture_weights'].log_weights(), axis=1)
        self.ess_mubar = np.exp(2*logsumexp(logw_folded) - logsumexp(2*logw_folded))
        # Compute ESJD for the first component of the mean
        fnk_weighted = attributes['pos_nk'][:, :, 0]*np.exp(attributes['logw'])
        differences = fnk_weighted[:, None, :] - fnk_weighted[:, :, None]
        squared_differences = np.sum(differences ** 2, axis=0)
        self.esjd_mean = np.triu(squared_differences, k=1).sum()
        self.esjd_mean /= np.exp(2 * logsumexp(attributes['logw']))

    def terminate(self) -> bool:
        """Terminates if the metric is less than or equal to the terminal metric.

        Parameters
        ----------
        :return: Whether termination is reached
        :rtype: bool
        """
        return self.grab_metric() <= self.terminal_metric


class MonitorMixtureIntSnippet:

    def __init__(self, *monitors: Monitor):
        """Monitors progress and computes metrics for an integrator snippets using a mixture of integrators.

        Parameters
        ----------
        :param monitors: Monitors for each integrator in the mixture
        :type monitors: Monitor
        """
        self.monitors = monitors

    def update_metrics(self, attributes: dict):
        """Updates metrics for each monitor.

        Parameters
        ----------
        :param attributes: Dictionary with the attributes of the integrator snippet
        :type attributes: dict
        """
        for ix in range(len(self.monitors)):
            # for each monitor, we require to pass a dictionary with the following keys
            # N, T, trajectory_indices, indices, particle_indices
            # of which N, T are common for all but the other are specific to each integrator/monitor
            keys = {'trajectory_indices', 'indices', 'particle_indices'}
            filtered_attributes = {k: v for k, v in attributes.items() if k not in keys}
            for key in keys:
                filtered_attributes[key] = attributes[key][attributes['iotas'] == ix]
            self.monitors[ix].update_metrics(filtered_attributes)

    def terminate(self) -> bool:
        """Terminates if any of its sub-monitors terminates.

        Parameters
        ----------
        :return: Whether termination is reached
        :rtype: bool
        """
        return np.any([monitor.terminate() for monitor in self.monitors])
