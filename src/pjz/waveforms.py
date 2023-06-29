"""Temporal waveforms for fdtdz."""

import jax.numpy as jnp


def ramped_sin(omega, width, delay, dt, tt):
  """Sine function with a gradual ramp.

  Based on MEEP's continuous source, see 
  https://meep.readthedocs.io/en/latest/FAQ/#why-doesnt-the-continuous-wave-cw-source-produce-an-exact-single-frequency-response

  """
  t = omega * dt * jnp.arange(tt)
  return ((1 + jnp.tanh(t / width - delay)) / 2) * jnp.sin(t)
