import pjz

import numpy as np
import pytest


@pytest.mark.parametrize("prop_axis", ["x", "y", "z"])
def test_mode_output_is_float32_and_correct_shape(prop_axis):
  uu, vv = 30, 20
  epsilon = np.ones((3, uu, vv))
  epsilon[:, 9:21, 8:12] = 12.25
  epsilon = np.expand_dims(epsilon, axis="xyz".find(prop_axis) + 1)
  omega = np.linspace(2 * np.pi / 37, 2 * np.pi / 31, 5)
  beta, field, err, iters = pjz.mode(
      epsilon=epsilon,
      omega=omega,
      num_modes=9,
      max_iters=10,  # Don't seriously attempt a solve.
  )
  assert beta.dtype == np.float32
  assert field.dtype == np.float32
  assert err.dtype == np.float32

  assert beta.shape == (5, 9)
  if prop_axis == "x":
    assert field.shape == (5, 2, 1, uu, vv, 9)
  elif prop_axis == "y":
    assert field.shape == (5, 2, uu, 1, vv, 9)
  else:  # prop_axis == "z".
    assert field.shape == (5, 2, uu, vv, 1, 9)
  assert err.shape == (5, 9)


@pytest.mark.parametrize("prop_axis", ["x", "y", "z"])
def test_correct_betas(prop_axis):
  expected_betas = (0.36388508, 0.18891069, 0.15406249, 0.13549446)
  omega = np.array([2 * np.pi / 37])
  uu, vv = 30, 20
  epsilon = np.ones((3, uu, vv))
  epsilon[:, 9:21, 8:12] = 12.25
  epsilon = np.expand_dims(epsilon, axis="xyz".find(prop_axis) + 1)
  beta, field, err, iters = pjz.mode(
      epsilon=epsilon,
      omega=omega,
      num_modes=len(expected_betas),
  )
  assert beta[0, :] == pytest.approx(expected_betas, rel=1e-3)


@pytest.mark.parametrize("prop_axis", ["x", "y", "z"])
def test_init(prop_axis):
  omega = np.array([2 * np.pi / 37])
  num_modes = 1
  uu, vv = 30, 20
  epsilon = np.ones((3, uu, vv))
  epsilon[:, 9:21, 8:12] = 12.25
  epsilon = np.expand_dims(epsilon, axis="xyz".find(prop_axis) + 1)

  # Initial solve.
  beta, field, _, _ = pjz.mode(
      epsilon=epsilon,
      omega=omega,
      num_modes=num_modes
  )

  # Updated solve.
  beta2, _, _, iters = pjz.mode(
      epsilon=epsilon,
      omega=omega,
      init=field,
      num_modes=num_modes,
  )
  np.testing.assert_array_almost_equal(beta, beta2, decimal=3)
  assert iters == 1


# TODO: Do this for other directions as well?
@pytest.mark.parametrize("prop_axis", ["x", "y", "z"])
def test_full_fields(prop_axis):
  ww, mm = 4, 2
  omega = np.linspace(2 * np.pi / 37, 2 * np.pi / 36, ww)
  uu, vv = 30, 20
  epsilon = np.ones((3, uu, vv))
  epsilon[:, 9:21, 8:12] = 12.25
  epsilon = np.expand_dims(epsilon, axis="xyz".find(prop_axis) + 1)
  # epsilon[0, 9:30, 10:19, 0] = 12.25
  # epsilon[1, 9:31, 10:18, 0] = 12.25
  # epsilon[2, 9:31, 10:19, 0] = 12.25
  beta, field, err, iters = pjz.mode(
      epsilon=epsilon,
      omega=omega,
      num_modes=mm,
  )

  if prop_axis == "x":
    f = field[:, :, 0, :, :, :]
    f = np.flip(f, axis=1)
    epsilon = epsilon[(1, 2, 0), ...]
    print("f shape", f.shape)
  elif prop_axis == "y":
    f = field[:, :, :, 0, :, :]
    f = np.flip(np.swapaxes(f, 2, 3), axis=2)
    epsilon = np.flip(np.swapaxes(epsilon[(2, 0, 1), ...], 1, 3), axis=1)
  else:  # prop_axis == "z".
    f = field[:, :, :, :, 0, :]
    f = (np.array([-1, 1])[None, :, None, None, None] * np.flip(f, axis=1))

  h, e, h2 = pjz._mode._full_fields(beta, omega, np.squeeze(epsilon), f)

  hflat, hflat2 = [x.reshape((ww, -1, mm)) for x in (h, h2)]
  np.testing.assert_array_less(
      np.linalg.norm(hflat - hflat2, axis=1) / np.linalg.norm(hflat, axis=1),
      1e-2)
  np.testing.assert_array_almost_equal(
      np.sum(e[:, 0] * h[:, 1] - e[:, 1] * h[:, 0], axis=(1, 2)), 1)
