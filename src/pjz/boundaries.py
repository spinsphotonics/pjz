"""Boundary conditions for fdtdz."""

import jax.numpy as jnp


def _absorption_profiles(numcells, width, smoothness):
  """1D quadratic profile for adiabatic absorption boundary conditions."""
  center = (numcells - 1) / 2
  offset = jnp.array([[0], [0.5]])
  pos = jnp.arange(numcells) + offset
  pos = jnp.abs(pos - center) - center + width
  pos = jnp.clip(pos, a_min=0, a_max=None)
  return smoothness * jnp.power(pos, 2)


def _cross_profiles(x, y):
  """Combine two 1D absorption profiles into a 2D profile."""
  return jnp.maximum(*jnp.meshgrid(x, y, indexing='ij'))[None, ...]


def absorption_mask(xx, yy, width, smoothness):
  """`(3, xx, yy)` absorption values in the x-y plane."""
  x = _absorption_profiles(xx, width, smoothness)
  y = _absorption_profiles(yy, width, smoothness)
  return jnp.concatenate([_cross_profiles(x[0], y[1]),
                          _cross_profiles(x[1], y[0]),
                          _cross_profiles(x[1], y[1])])


def _safe_div(x, y):
  return jnp.zeros_like(x) if y == 0 else x / y


def pml_sigma(pml_widths, zz, ln_R=16.0, m=4.0):
  """`(zz, 2)` array of conductivity values for PML along the z-axis."""
  offset = jnp.array([[0], [0.5]])
  z = jnp.arange(zz) + offset
  z = jnp.stack([_safe_div(pml_widths[0] - z, pml_widths[0]),
                 _safe_div(z + 0.5 - zz + pml_widths[1], pml_widths[1])],
                axis=-1)
  z = jnp.max(jnp.clip(z, a_min=0, a_max=None), axis=-1)
  return ((m + 1) * ln_R * z**m).T
