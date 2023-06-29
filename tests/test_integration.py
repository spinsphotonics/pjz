import pjz

import jax.numpy as jnp
import numpy as np
import pytest


def get_beta_xy(i, x, y):
  xx, yy = 30, 20
  pos = (np.arange(2 * xx)[:, None], np.arange(2 * yy))
  eps = 1 + 12.25 * pjz.rect(pos, (xx + x, yy + y), (25, 10))
  epsilon = pjz.render(eps[None, :, :], np.array([]),
                       np.zeros((1, 2)), np.ones((1, 2)), m=1)
  beta, _ = pjz.waveguide(i, 2 * np.pi / 37, np.array(epsilon[:, :, :, 0]),
                          np.ones((xx, 2)), np.ones((yy, 2)))
  return beta


@pytest.mark.parametrize("i,dx,dy,expected", [
    (0, 0.2, 0.2, 1e-2),
])
def test_spread_xy(i, dx, dy, expected):
  betas = [get_beta_xy(i, x, y) for x in np.arange(0, 1, dx)
           for y in np.arange(0, 1, dy)]
  max_var = (np.max(betas) - np.min(betas)) / 2 / np.mean(betas)
  assert max_var <= expected


def get_beta_xz(i, x, z):
  xx, yy, zz = 30, 1, 20
  pos = (np.arange(2 * xx)[:, None], np.arange(2 * yy))
  eps = np.ones((3, 2 * xx, 2 * yy))
  eps[1, ...] = 1 + 12.25 * pjz.rect(pos, (xx + x, yy), (25, np.inf))
  epsilon = pjz.render(eps, np.array([7.5, 12.5]) + z,
                       np.arange(zz)[:, None] + [-0.5, 0],
                       np.arange(zz)[:, None] + [0.5, 1.0], m=1)
  beta, _ = pjz.waveguide(i, 2 * np.pi / 37, np.array(epsilon[:, :, 0, :]),
                          np.ones((xx, 2)), np.ones((zz, 2)))
  return beta


@pytest.mark.parametrize("i,dx,dz,expected", [
    (0, 0.2, 0.2, 1e-2),
])
def test_spread_xz(i, dx, dz, expected):
  betas = [get_beta_xz(i, x, z) for x in np.arange(0, 1, dx)
           for z in np.arange(0, 1, dz)]
  max_var = (np.max(betas) - np.min(betas)) / 2 / np.mean(betas)
  assert max_var <= expected
