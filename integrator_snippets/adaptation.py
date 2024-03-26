import numpy as np


class AdaptationStrategy:

    def __init__(self):
        """Implements adaptation strategies for the parameters of integrators."""
        pass

    def adapt(self, attributes: dict) -> dict:
        """Adapts the parameters of the integrator(s).

        Parameters
        ----------
        :param attributes: Dictionary with the attributes of the integrator snippet
        :type attributes: dict
        :return: Dictionary with the adapted parameters
        :rtype: dict
        """
        raise NotImplementedError


class DummyAdaptation(AdaptationStrategy):

    def __init__(self):
        """Dummy adaptation strategy, does nothing, but allows to maintain a simple interface throughout."""
        super().__init__()

    def adapt(self, attributes: dict) -> dict:
        """Does nothing.

        Parameters
        ----------
        :param attributes: Dictionary with the attributes of the integrator snippet
        :type attributes: dict
        :return dict: Dictionary with the same step size as before, i.e. does nothing
        :rtype: dict
        """
        return {'step_size': attributes['integrator'].__dict__['step_size']}


class SingleStepSizeAdaptorSA(AdaptationStrategy):

    def __init__(self, target_metric_value: float, metric: str = 'pm', lr: float = 0.5, min_step: float = 1e-30,
                 max_step: float = 100.0):
        """Adapts step size of an integrator snippet with a single integrator, using stochastic approximations.

        Parameters
        ----------
        :param target_metric_value: Target value for the metric, used for adaptation
        :type target_metric_value: float
        :param metric: Metric to use for adaptation. Must be one of 'pm', 'mip', or 'mpd'.
        :type metric: str
        :param lr: Learning rate for the adaptation
        :type lr: float
        :param min_step: Minimum step size allowed
        :type min_step: float
        :param max_step: Maximum step size allowed
        :type max_step: float

        Notes
        -----
        It adapts only the step size of the integrator and it does so using adaptive monte carlo methods of
        Andrieu (2008), specifically stochastic approximations.
        """
        super().__init__()
        assert metric in {'pm', 'mip', 'mpd'}, "Metric must be one of 'pm', 'mip', or 'mpd'."
        self.lr = lr
        self.min_step = min_step
        self.max_step = max_step
        self.target_metric_value = target_metric_value
        if metric == 'pm':
            self.metric_key = "proportion_moved"
        elif metric == 'mip':
            self.metric_key = "median_index_proportion"
        else:
            self.metric_key = "median_path_diversity"

    def adapt(self, attributes: dict) -> dict:
        """Performs adaptation on the step size.

        Parameters
        ----------
        :param attributes: Dictionary with the attributes of the integrator snippet
        :type attributes: dict
        :return: Dictionary with the adapted step size
        :rtype: dict
        """
        log_step = np.log(attributes['integrator'].__dict__['step_size'])
        metric_value = attributes['monitor'].__dict__[self.metric_key]
        return {
            'step_size': np.clip(
                np.exp(log_step + self.lr*(metric_value - self.target_metric_value)),
                a_min=self.min_step,
                a_max=self.max_step)
        }


class MixtureStepSizeAdaptorSA:

    def __init__(self, *adaptors: AdaptationStrategy):
        """Adapts the step size of a mixture of integrators using stochastic approximations.

        Parameters
        ----------
        :param adaptors: Adaptors for the individual integrators
        :type adaptors: AdaptationStrategy
        """
        self.adaptors = adaptors

    def adapt(self, attributes: dict):
        """Adapts the step size of each integrator.

        Parameters
        ----------
        :param attributes: Dictionary with the attributes of the integrator snippet
        :type attributes: dict
        :return: Dictionary with the adapted step sizes, for each integrator
        :rtype: dict
        """
        adaptation_dict = {ix: {} for ix in range(len(self.adaptors))}
        for ix in range(len(self.adaptors)):
            # creates a dict that contains 'integrator' and 'monitor'
            filtered_attributes = {
                'integrator': attributes['integrators'].integrators[ix],
                'monitor': attributes['monitors'].monitors[ix]
            }
            adaptation_dict[ix] = self.adaptors[ix].adapt(filtered_attributes)
        return adaptation_dict


class SingleStepSizeSATMinAdaptor(AdaptationStrategy):

    def __init__(self, target_metric_value: float, metric: str = 'pm', lr: float = 0.5, min_step: float = 1e-30,
                 max_step: float = 100.0):
        super().__init__()
        self.target_metric_value = target_metric_value
        self.metric = metric
        self.lr = lr
        self.min_step = min_step
        self.max_step = max_step
        if metric == 'pm':
            self.metric_key = "proportion_moved"
        elif metric == 'mip':
            self.metric_key = "median_index_proportion"
        else:
            self.metric_key = "median_path_diversity"

    def adapt(self, attributes: dict) -> dict:
        """Adapts the step size of the integrator using stochastic approximation and T based on the
        median index proportion."""
        # Stochastic Approximation
        log_step = np.log(attributes['integrator'].__dict__['step_size'])
        metric_value = attributes['monitor'].__dict__[self.metric_key]
        return {
            'step_size': np.clip(
                np.exp(log_step + self.lr*(metric_value - self.target_metric_value)),
                a_min=self.min_step,
                a_max=self.max_step),
            'T': np.max(attributes['k_resampled'])
        }
