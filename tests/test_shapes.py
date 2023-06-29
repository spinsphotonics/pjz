from pjz import shapes

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_fundamental_ops():
  assert shapes.invert(1) == 0
  assert shapes.invert(0) == 1
  assert shapes.invert(0.5) == 0.5

  assert shapes.union(0.5, 0) == 0.5
  assert shapes.union(1, 0) == 1
  assert shapes.union(1, 0.5) == 1
  assert shapes.union(0, 0.5) == 0.5

  assert shapes.intersect(0.5, 1) == 0.5
  assert shapes.intersect(0, 1) == 0
  assert shapes.intersect(1, 0) == 0
  assert shapes.intersect(1, 0.5) == 0.5

  assert jax.grad(shapes.invert)(0.0) == -1
  assert jax.grad(shapes.union, (0, 1))(0.5, 0.6) == (0, 1)
  assert jax.grad(shapes.intersect, (0, 1))(0.5, 0.6) == (1, 0)


def test_rect():
  np.testing.assert_array_equal(
      shapes.rect((jnp.arange(3)[:, None], jnp.arange(3)), (1, 1), (2, 2)),
      [[0.25, 0.50, 0.25],
       [0.50, 1.00, 0.50],
       [0.25, 0.50, 0.25],])

  grad_center, grad_widths = jax.grad(
      lambda center, widths:
          jnp.sum(shapes.rect((jnp.arange(3)[:, None], jnp.arange(3)),
                              center, widths)),
      (0, 1))((1.0, 1.0), (2.0, 2.0))

  assert grad_center == (0, 0)
  assert grad_widths == (2, 2)


def test_circ():
  foo = 1.5 - np.sqrt(2)
  np.testing.assert_array_almost_equal(
      shapes.circ((jnp.arange(3)[:, None], jnp.arange(3)), (1, 1), 1),
      [[foo, 0.5, foo],
       [0.5, 1.0, 0.5],
       [foo, 0.5, foo],])

  grad_center, grad_radius = jax.grad(
      lambda center, radius:
          jnp.sum(shapes.circ((jnp.arange(3)[:, None], jnp.arange(3)),
                              center, radius)),
      (0, 1))((1.0, 1.0), 1.0)

  assert grad_center == (0, 0)
  assert grad_radius == 8

  # Test the internal implementation of the safe sqrt gradient.
  assert jax.grad(shapes._sqrt)(0.42) == jax.grad(jnp.sqrt)(0.42)
  assert jax.grad(shapes._sqrt)(0.0) == 0.0
