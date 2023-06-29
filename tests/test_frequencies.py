import pjz

import jax.numpy as jnp
import numpy as np
import pytest


def test_sampling_interval():
  wmin, wmax = 2 * np.pi/40, 2 * np.pi / 36
  n = 10
  dt = 0.5
  interval = pjz.sampling_interval(wmin, wmax, n, dt)

  ws = np.linspace(wmin, wmax, n)
  theta = ws * dt * interval * np.arange(2 * n)[:, None]
  A = np.hstack([np.sin(theta), np.cos(theta)])
  A /= np.linalg.norm(A, ord=2, axis=0)
  assert np.all(A.T @ A - np.eye(2 * n) < 1e-1)


def test_frequency_components():
  omegas = jnp.array([2 * jnp.pi / 40, 2 * jnp.pi / 39])
  steps = jnp.array([10, 20, 30, 40])
  dt = 0.5
  signal = jnp.sin(omegas[0] * dt * steps) + jnp.cos(omegas[1] * dt * steps)
  out = pjz.frequency_components(
      signal[:, None, None, None, None], steps, omegas, dt)
  assert out[0] == pytest.approx(-1j, rel=1e-3)
  assert out[1] == pytest.approx(1, rel=1e-3)


def test_source_amplitude():
  omega = 2 * jnp.pi / 40
  dt = 0.5
  tt = 10000
  out = pjz.source_amplitude(
      jnp.sin(omega * dt * (jnp.arange(tt) - 0.5))[:, None], omega, dt)
  assert out[0] == pytest.approx(-1j)
