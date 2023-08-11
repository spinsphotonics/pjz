"""Basic shape operations."""

import jax
import jax.numpy as jnp

from typing import Tuple
from math import prod


def rect(
        shape: Tuple[int, ...],
        center: jax.Array,
        widths: jax.Array,
) -> jax.Array:
  """Rectangle."""
  # return prod(jnp.clip((w + 1) / 2 - jnp.abs(p - c), 0, 1)
  #             for p, c, w in zip(pos, center, widths))


@jax.custom_jvp
def _sqrt(x):
  return jnp.sqrt(x)


_sqrt.defjvps(lambda t, ans, x: 0.5 /
              jnp.where(x == 0, jnp.inf, jnp.sqrt(x)) * t)


def circ(
        shape: Tuple[int, ...],
        center: jax.Array,
        radius: jax.Array,
) -> jax.Array:
  """Circle."""
  # dist = _sqrt(sum((p - c)**2 for p, c in zip(pos, center)))
  # return jnp.clip((radius + 0.5) - dist, 0, 1)


def invert(a):
  return 1 - a


def union(a, b):
  return jnp.maximum(a, b)


def intersect(a, b):
  return jnp.minimum(a, b)


def dilate(a: jax.Array, radius: float) -> jax.Array:
  """Dilate"""


def shift(a: jax.Array, axis: int, distance: float) -> jax.Array:
  """Shift."""
