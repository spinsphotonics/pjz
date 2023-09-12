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
  """Computes a per-pixel density from raw optimization variable ``u``.

  Implements the "three-field" scheme detailed in [#threefield_ref]_ in order
  to allow for a final density that is binary (with the exception of boundary
  values) and that conforms to a minimum feature size requirement.

  .. [#threefield_ref] Zhou, Mingdong, et al. "Minimum length scale in topology
    optimization by geometric constraints." Computer Methods in Applied Mechanics 
    and Engineering 293 (2015): 266-282.

  Args:
    u: `(xx, yy)`` variable array with values within ``[0, 1]``.
    radius: Radius of the conical filter used to blur ``u``.
    alpha: Float within ``[0, 1]`` controlling binarization of the density,
      where ``0`` denotes no binarization, and ``1`` denotes full binarization
      of all pixels except those on boundaries (as given by ``eta``) which are
      left unchanged (equal to the ``alpha = 0`` case).
    c: Controls the detection of inflection points.
    eta: Threshold value used to binarize the density.
    eta_lo: Controls minimum feature size of void-phase features.
    eta_hi: Controls minimum feature size of ``density = 1`` features.

  Returns:
    ``(density, loss)`` arrays, both of shape ``(xx, yy)``, corresponding to
    the pixel density and the minimum feature size loss respectively.

  """
  ufilt = _filter(u, radius)
  uproj = _project(ufilt, eta)
  return (alpha * uproj + (1 - alpha) * ufilt,
          _geom_loss(uproj, ufilt, c, eta_lo, eta_hi))
