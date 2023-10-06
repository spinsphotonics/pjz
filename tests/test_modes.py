import pjz

import numpy as np
import pytest


def test_mode_output_is_float32():
  xx, yy = 40, 20
  epsilon = np.ones((3, xx, yy, 1))
  epsilon[:, 9:31, 8:12, 0] = 12.25
  beta, field, err, iters = pjz.mode(
      epsilon=epsilon,
      omega=(2 * np.pi / 37),
      num_modes=1,
  )
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
  epsilon = np.ones((3, xx, yy, 1))
  epsilon[:, 9:21, 8:12, 0] = 12.25
  beta, field, err, iters = pjz.mode(
      epsilon=epsilon,
      omega=(2 * np.pi / 37),
      num_modes=i + 1,
  )
  assert beta[i] == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_check_it(i):
  xx, yy = 40, 30
  omega = (2 * np.pi / 37)
  epsilon = np.ones((3, xx, yy, 1))
  epsilon[0, 9:30, 10:19, 0] = 12.25
  epsilon[1, 9:31, 10:18, 0] = 12.25
  epsilon[2, 9:31, 10:19, 0] = 12.25
  beta, field, err, iters = pjz.mode(
      epsilon=epsilon,
      omega=omega,
      num_modes=i + 1,
  )
  f = np.array([-1, 1])[:, None, None, None] * field[(1, 0), ..., i]
  h, e, h2 = pjz._mode._check_it(beta[i], omega, epsilon, f)
  assert np.linalg.norm(h - h2) / np.linalg.norm(h) < 1e-2


# def test_no_propagating_mode():
#   xx, yy = 30, 20
#   epsilon = np.ones((3, xx, yy))
#   epsilon[:, 9:21, 8:12] = 12.25
#   with pytest.raises(ValueError, match="No propagating mode found"):
#     beta, field = pjz.waveguide(4, 2 * np.pi / 37, epsilon,
#                                 np.ones((xx, 2)), np.ones((yy, 2)))
#
#
# @pytest.mark.parametrize("i", [0, 1, 2, 3])
# def test_double_curl(i):
#   xx, yy = 30, 20
#   epsilon = 1.5 * np.ones((3, xx, yy))
#   epsilon[:, 9:21, 8:12] = 12.25
#   omega = 2 * np.pi / 37
#   dx, dy = np.ones((xx, 2)), np.ones((yy, 2))
#   beta, field = pjz.waveguide(i, omega, epsilon, dx, dy)
#   e2h, h2e = pjz.modes._conversion_operators(beta, omega, epsilon, dx, dy)
#
#   hfields = np.reshape(e2h @ np.ravel(field), field.shape)
#   efields = np.reshape(h2e @ np.ravel(hfields), field.shape)
#
#   # Test that we can re-convert back to E-fields with the same result.
#   np.testing.assert_array_almost_equal(efields, field)
#
#   # Test that the pwoer in the mode is `1`.
#   assert pjz.modes._power_in_mode(
#       field, beta, omega, epsilon, dx, dy) == pytest.approx(1.0)
