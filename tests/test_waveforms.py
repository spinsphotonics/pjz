import pjz

import jax.numpy as jnp
import numpy as np
import pytest


def test_ramped_sin():
  omega = 2 * jnp.pi / 40
  dt = 0.5
  tt = 100000
  np.testing.assert_array_almost_equal(
      pjz.ramped_sin(omega, width=3, delay=4, dt=dt, tt=tt)[-10:],
      jnp.sin(omega * dt * jnp.arange(tt))[-10:])
