"""Per-pixel density."""

from functools import partial
import math

import jax
import jax.numpy as jnp


def _cone(radius):
  r = math.ceil(radius - 0.5)
  u = jnp.square(jnp.array([i - r for i in range(2 * r + 1)]))
  weights = jnp.maximum(0, radius - jnp.sqrt(u + u[:, None]))
  return weights / jnp.sum(weights)


def _filter(u, radius):
  cone = _cone(radius)
  u = jnp.pad(u, [(s // 2, s // 2) for s in cone.shape], mode="edge")
  return jax.scipy.signal.convolve(u, cone, mode="valid")


def _boundary_cell(u):
  u = jnp.pad(u >= 0, 1, mode="edge")
  u = [jnp.logical_xor(u, jnp.roll(u, shift, axis))[1:-1, 1:-1]
       for shift in (1, -1) for axis in (0, 1)]
  return jnp.any(jnp.stack(u), axis=0)


def _project(u, eta):
  return jnp.where(_boundary_cell(u - eta), u, (jnp.sign(u - eta) + 1) / 2)


def _inflection(u, c):
  u = jnp.pad(u, 1, "edge")
  gu2 = (jnp.square(u[2:, 1:-1] - u[:-2, 1:-1]) +
         jnp.square(u[1:-1, 2:] - u[1:-1, :-2]))
  return jnp.exp(-c * gu2)


def _geom_loss(uproj, ufilt, c, eta_lo, eta_hi):
  uinfl = _inflection(ufilt, c)
  return (uproj * uinfl * jnp.square(jnp.minimum(0, ufilt - eta_hi)) +
          (1 - uproj) * uinfl * jnp.square(jnp.minimum(0, eta_lo - ufilt)))


@partial(jax.jit, static_argnames=["radius"])
def density(u, radius, alpha, c=1.0, eta=0.5, eta_lo=0.25, eta_hi=0.75):
  ufilt = _filter(u, radius)
  uproj = _project(ufilt, eta)
  return (alpha * uproj + (1 - alpha) * ufilt,
          _geom_loss(uproj, ufilt, c, eta_lo, eta_hi))
