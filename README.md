![Integrator Snippets](integrator_snippets.png)

# Integrator Snippets
Integrator Snippets are a novel class of algorithms to sample from a target distribution.

## Structure

---
### `integrator_snippets.py`
Classes for **unfolded** integrator snippets.
- `SingleIntegratorSnippet`: trajectories are constructed with a single integrator $\psi:\mathsf{Z}\to\mathsf{Z}$ with step size $\delta>0$.
- `MixtureIntegratorSnippetSameT`: trajectories are constructed using multiple integrators $\psi_1, \ldots, \psi_I:\mathsf{Z}\to\mathsf{Z}$. All integrators are run for the same number of steps $T\in\mathbb{Z}_+$ but possibly different step sizes.
---
### `integrators.py`
Classes to define integrators.
- `Integrator`: use it to create a new integrator, which can then be used either as $\psi$ in `SingleIntegratorSnippet` or as one of the integrators in `MixtureIntegratorSnippetSameT`. Two examples are already implemented:
  - `LeapfrogIntegrator`: classic HMC Leapfrog integrator.
  - `AMIntegrator`: THUG and SNUG integrators (or a mixture of them).
- `IntegratorMixtureSameT`: class used to collect together a bunch of integrators, so that they can be passed to `MixtureIntegratorSnippetSameT`. For instance, one may wish to create a THUG integrator and a SNUG integrator and use them jointly.
---
### `distributions.py`
Classes to define target distributions. Filamentary distributions

$$
\pi_\epsilon(dx) \propto k_\epsilon(f(x)) \pi(dx), \qquad \epsilon > 0
$$

where $\pi(dx)$ is a distribution on $(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d))$, $f:\mathbb{R}^n\to\mathbb{R}^m$ is a smooth function with $n > m$, $k_\epsilon:\mathbb{R}\to\mathbb{R}_+$ is a kernel function (or more appropriately an approximation to the identity) with tolerance $\epsilon>0$ are fully implemented.
Tempered distributions are still under development.

---
### `monitoring.py`
Classes to monitor the performance and track the execution/termination of integrator snippets. 
Computes metrics such as the proportion of particles moved, the median index proportion and many others. It has been built to be very flexible.
---
### `adaptation.py`
Classes to adapt parameters of integrators (both single and mixtures of integrators).


