![Integrator Snippets](integrator_snippets.png)

# Integrator Snippets
Integrator Snippets are a novel class of algorithms to sample from a target distribution.

## Structure

- `integrator_snippets.py`: Classes for **unfolded** integrator snippets.
    - `SingleIntegratorSnippet`: trajectories are constructed with a single integrator $\psi:\mathsf{Z}\to\mathsf{Z}$ with step size $\delta>0$.
    - `MixtureIntegratorSnippetSameT`: trajectories are constructed using multiple integrators $\psi_1, \ldots, \psi_I:\mathsf{Z}\to\mathsf{Z}$. All integrators are run for the same number of steps $T\in\mathbb{Z}_+$ but possibly different step sizes.

- `integrators.py`: contains classes to define integrators. Two main classes are available:
  - `Integrator`: this should be used to create a new integrator, which can then be used either as $\psi$ in `SingleIntegratorSnippet` or as one of the integrators in `MixtureIntegratorSnippetSameT`. Two examples are already implemented:
    - `LeapfrogIntegrator`: classic HMC Leapfrog integrator.
    - `AMIntegrator`: THUG and SNUG integrators (or a mixture of them).
  - `IntegratorMixtureSameT`: class used to collect together a bunch of integrators, so that they can be passed to `MixtureIntegratorSnippetSameT`. For instance, one may wish to create a THUG integrator and a SNUG integrator and use them jointly.
- `distributions.py`: contains classes to define target distributions. Currently Filamentary distributions of the form $\pi_\epsilon(dx) \propto k_\epsilon(f(x)) \pi(dx)$ are fully implemented, whereas Tempered distributions are still under development.
- `monitoring.py`: classes to monitor the performance and track the execution/termination of integrator snippets. Computes metrics such as the proportion of particles moved, the median index proportion and many others. It has been built to be very flexible.
- `adaptation.py`: classes to adapt parameters of integrators (both single and mixtures of integrators).


