# Integrator Snippets
Integrator Snippets are a novel class of algorithms to sample from a target distribution.

### Repo Structure

- `integrator_snippets.py` contains the main code implementing the actual integrator snippet algorithms. Two variants are currently available:
    - `SingleIntegratorSnippet` uses a single integrator $\psi$.
    - `MixtureIntegratorSnippetSameT`: allows the user to specify a finite number of integrators, all with the same number of integration steps (but possibly different step sizes). 

- `integrators.py`: contains classes to define integrators. Two main classes are available:
  - `Integrator`: this should be used to create a new integrator, which can then be used either as $\psi$ in `SingleIntegratorSnippet` or as one of the integrators in `MixtureIntegratorSnippetSameT`. Two examples are already implemented:
    - `LeapfrogIntegrator`: implements the classic HMC Leapfrog integrator.
    - `AMIntegrator`: implements THUG and SNUG integrators (or a mixture of them).
  - `IntegratorMixtureSameT`: class used to collect together a bunch of integrators, so that they can be passed to `MixtureIntegratorSnippetSameT`. For instance, one may wish to create a THUG integrator and a SNUG integrator and use them jointly.
- `distributions.py`: contains classes to define target distributions. Currently Filamentary distributions are fully implemented, whereas Tempered distributions are still under development.
- `monitoring.py`: classes to monitor the performance and track the execution/termination of integrator snippets. Computes metrics such as the proportion of particles moved, the median index proportion and many others. It has been built to be very flexible.
- `adaptation.py`: classes to adapt parameters of integrators (both single and mixtures of integrators).


