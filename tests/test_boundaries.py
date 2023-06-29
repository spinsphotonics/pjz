import pjz

import jax.numpy as jnp
import numpy as np
import pytest


def test_absorption_mask():
  abs_mask = pjz.absorption_mask(xx=10, yy=10, width=3, smoothness=4)

  np.testing.assert_array_equal(
      abs_mask[0, :, 5], [36, 16, 4, 0, 0, 0, 0, 4, 16, 36])
  np.testing.assert_array_equal(
      abs_mask[0, 5, :], [25, 9, 1, 0, 0, 0, 1, 9, 25, 49])

  np.testing.assert_array_equal(
      abs_mask[1, :, 5], [25, 9, 1, 0, 0, 0, 1, 9, 25, 49])
  np.testing.assert_array_equal(
      abs_mask[1, 5, :], [36, 16, 4, 0, 0, 0, 0, 4, 16, 36])

  np.testing.assert_array_equal(
      abs_mask[2, :, 5], [25, 9, 1, 0, 0, 0, 1, 9, 25, 49])
  np.testing.assert_array_equal(
      abs_mask[2, 5, :], [25, 9, 1, 0, 0, 0, 1, 9, 25, 49])


def test_pml_sigma():
  print(pjz.pml_sigma(pml_widths=(4, 4), zz=10, ln_R=16.0, m=4.0))
  np.testing.assert_array_almost_equal(
      pjz.pml_sigma(pml_widths=(4, 4), zz=10, ln_R=16.0, m=4.0),
      [[8.0000000e+01, 4.6894531e+01],
       [2.5312500e+01, 1.2207031e+01],
       [5.0000000e+00, 1.5820312e+00],
       [3.1250000e-01, 1.9531250e-02],
       [0.0000000e+00, 0.0000000e+00],
       [0.0000000e+00, 0.0000000e+00],
       [1.9531250e-02, 3.1250000e-01],
       [1.5820312e+00, 5.0000000e+00],
       [1.2207031e+01, 2.5312500e+01],
       [4.6894531e+01, 8.0000000e+01]]
  )
