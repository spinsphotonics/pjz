"""Back out frequency components."""

import jax.numpy as jnp
import numpy as np


def frequency_components(out, steps, omegas, dt):
  """Returns E-field at `omega` for simulation output `out` at `steps`."""
  n = len(omegas)
  theta = omegas[:, None] * dt * steps
  phases = jnp.concatenate([jnp.cos(theta), -jnp.sin(theta)], axis=0)
  parts = jnp.einsum('ij,jk...->ik...', jnp.linalg.pinv(phases.T), out)
  return parts[:n] + 1j * parts[n:]


def source_amplitude(source_waveform, omega, dt):
  """Returns complex scalar denoting source amplitude at `omega`."""
  # `0.5` offset accounts for the E-field source being colocated in time with
  # the H-field updates.
  theta = omega * dt * (jnp.arange(source_waveform.shape[0]) - 0.5)

  # Extracts the complex amplitude of the current source at `omega`.
  parts = jnp.mean((2 *
                    jnp.stack([jnp.cos(theta), -jnp.sin(theta)])[..., None] *
                    source_waveform),
                   axis=1)
  return parts[0] + 1j * parts[1]


def sampling_interval(wmin, wmax, n, dt):
  """Sampling interval to observe `n` components within `[wmin, wmax]`."""
  # First, we find the interval at which the maximum and minimum angular
  # frequencies generate a phase difference of `pi * n / (n - 1)`.
  cutoff = np.pi * (n - 1) / n / (wmax - wmin) / dt

  # Now, we use the the periodicity of the center frequency to find the nearest
  # "open window" (a time step in which no frequency component will complete a
  # rotation which is a multiple of `pi` -- since this would make this component
  # unobservable), and use this as our sampling interval.
  period = 4 * np.pi / (wmax + wmin) / dt
  m = np.floor((cutoff - period / 4) / (period / 2))
  return int(round(period / 4 + m * period / 2))
