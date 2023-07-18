"""Frequency-related operations."""


import jax
import jax.numpy as jnp

from typing import Tuple


def ramped_sin(
        omega: jax.Array,
        dt: Union[float, jax.Array],
        tt: int,
        width: Union[float, jax.Array] = 4,
        delay: Union[float, jax.Array] = 4,
) -> jax.Array:
  """Sine function with a gradual ramp.

  Based on MEEP's continuous source, see 
  https://meep.readthedocs.io/en/latest/FAQ/#why-doesnt-the-continuous-wave-cw-source-produce-an-exact-single-frequency-response

  Args:
    omega: Array of angular frequencies.
    dt: Dimensionless time step.
    tt: Number of update steps.
    width: Number of periods in ramp-up time.
    delay: Number of periods until mid-ramp.

  Returns:
    Array with ``shape[-1] == tt`` and ``shape[:-1]`` equal to the input 
    parameter shapes broadcast together.
    
  """
  dt, width, delay = [jnp.array(arr)[..., None] for arr in (dt, width, delay)]
  t = omega[..., None] * dt * jnp.arange(tt)
  return ((1 + jnp.tanh(t / width - delay)) / 2) * jnp.sin(t)


def sampling_interval(
        omega_min: float,
        omega_max: float,
        omega_n: int,
        dt: float
) -> int:
  """Sampling interval to observe `n` components within `[wmin, wmax]`."""
  # # First, we find the interval at which the maximum and minimum angular
  # # frequencies generate a phase difference of `pi * n / (n - 1)`.
  # cutoff = np.pi * (n - 1) / n / (wmax - wmin) / dt

  # # Now, we use the the periodicity of the center frequency to find the nearest
  # # "open window" (a time step in which no frequency component will complete a
  # # rotation which is a multiple of `pi` -- since this would make this component
  # # unobservable), and use this as our sampling interval.
  # period = 4 * np.pi / (wmax + wmin) / dt
  # m = np.floor((cutoff - period / 4) / (period / 2))
  # return int(round(period / 4 + m * period / 2))


def output_transform(
        omega: jax.Array,
        steps: Tuple[int, ...],
        dt: float,
) -> jax.Array:
  """Output transform."""
  # """Returns E-field at `omega` for simulation output `out` at `steps`."""
  # n = len(omegas)
  # theta = omegas[:, None] * dt * steps
  # phases = jnp.concatenate([jnp.cos(theta), -jnp.sin(theta)], axis=0)
  # parts = jnp.einsum('ij,jk...->ik...', jnp.linalg.pinv(phases.T), out)
  # return parts[:n] + 1j * parts[n:]
