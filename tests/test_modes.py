import pjz

import numpy as np
import pytest


def test_float32():
  xx, yy = 30, 20
  epsilon = np.ones((3, xx, yy))
  epsilon[:, 9:21, 8:12] = 12.25
  beta, field = pjz.waveguide(0, 2 * np.pi / 37, epsilon,
                              np.ones((xx, 2)), np.ones((yy, 2)))
  assert beta.dtype == np.float32
  assert field.dtype == np.float32


@pytest.mark.parametrize("i,expected", [
    (0, 0.36388508),
    (1, 0.18891069),
    (2, 0.15406249),
    (3, 0.13549446),
])
def test_find_pjz(i, expected):
  xx, yy = 30, 20
  epsilon = np.ones((3, xx, yy))
  epsilon[:, 9:21, 8:12] = 12.25
  beta, field = pjz.waveguide(i, 2 * np.pi / 37, epsilon,
                              np.ones((xx, 2)), np.ones((yy, 2)))
  assert beta == pytest.approx(expected)


def test_no_propagating_mode():
  xx, yy = 30, 20
  epsilon = np.ones((3, xx, yy))
  epsilon[:, 9:21, 8:12] = 12.25
  with pytest.raises(ValueError, match="No propagating mode found"):
    beta, field = pjz.waveguide(4, 2 * np.pi / 37, epsilon,
                                np.ones((xx, 2)), np.ones((yy, 2)))


@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_double_curl(i):
  xx, yy = 30, 20
  epsilon = 1.5 * np.ones((3, xx, yy))
  epsilon[:, 9:21, 8:12] = 12.25
  omega = 2 * np.pi / 37
  dx, dy = np.ones((xx, 2)), np.ones((yy, 2))
  beta, field = pjz.waveguide(i, omega, epsilon, dx, dy)
  e2h, h2e = pjz.modes._conversion_operators(beta, omega, epsilon, dx, dy)

  hfields = np.reshape(e2h @ np.ravel(field), field.shape)
  efields = np.reshape(h2e @ np.ravel(hfields), field.shape)

  # Test that we can re-convert back to E-fields with the same result.
  np.testing.assert_array_almost_equal(efields, field)

  # Test that the pwoer in the mode is `1`.
  assert pjz.modes._power_in_mode(
      field, beta, omega, epsilon, dx, dy) == pytest.approx(1.0)
