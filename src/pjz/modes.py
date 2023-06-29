"""Modes for finite-difference Yee-cell grids."""

import numpy as np
import scipy.sparse as sp


def _dx0(dx, dy):
  xx, yy = dx.shape[0], dy.shape[0]
  return sp.diags(
      [np.repeat(u, yy) for u in (dx[:, 0], -dx[1:, 0], -dx[0, 0])],
      [i * yy for i in (0, -1, xx - 1)])


def _dx1(dx, dy):
  xx, yy = dx.shape[0], dy.shape[0]
  return sp.diags(
      [np.repeat(u, yy) for u in (dx[:, 1], -dx[:-1, 1], -dx[-1, 1])],
      [i * yy for i in (0, 1, -(xx - 1))])


def _dy0(dx, dy):
  xx, yy = dx.shape[0], dy.shape[0]
  return sp.block_diag(
      [sp.diags((dy[:, 0], -dy[1:, 0], -dy[0, 0]), (0, -1, yy - 1))] * xx)


def _dy1(dx, dy):
  xx, yy = dx.shape[0], dy.shape[0]
  return sp.block_diag(
      [sp.diags((dy[:, 1], -dy[:-1, 1], -dy[-1, 1]), (0, 1, -(yy - 1)))] * xx)


def _waveguide_operator(omega, epsilon, dx, dy):
  """Waveguide operator as a sparse scipy matrix."""
  # Note: We assume row-major -- `flatindex = y + yy * x`.
  dx0, dx1, dy0, dy1 = _dx0(dx, dy), _dx1(dx, dy), _dy0(dx, dy), _dy1(dx, dy)
  exy = sp.diags(np.ravel(epsilon[:2]))
  inv_ez = sp.diags(1 / np.ravel(epsilon[2]))
  return (omega**2 * exy -
          sp.vstack([dx1, dy1]) @ inv_ez @ sp.hstack([dx0, dy0]) @ exy -
          sp.vstack([-dy0, dx0]) @ sp.hstack([-dy1, dx1]))


def _find_largest_eigenvalue(A, numsteps):
  """Estimate dominant eigenvector using power iteration."""
  v = np.random.rand(A.shape[0])
  for _ in range(numsteps):
    v = A @ v
    v /= np.linalg.norm(v)
  return v @ A @ v


def _conversion_operators(beta, omega, epsilon, dx, dy):
  """Operators for converting `[Ex, Ey]` to `[Hx, Hy]` and back."""
  dx0, dx1, dy0, dy1 = _dx0(dx, dy), _dx1(dx, dy), _dy0(dx, dy), _dy1(dx, dy)
  exy = sp.diags(np.ravel(epsilon[:2]))
  inv_exy = sp.diags(1 / np.ravel(epsilon[:2]))
  inv_ez = sp.diags(1 / np.ravel(epsilon[2]))
  eye = sp.eye(dx.shape[0] * dy.shape[0])
  foo = sp.bmat([[0 * eye, 1 * eye], [-1 * eye, 0 * eye]])
  e2h = -(sp.vstack([dy1, -dx1]) @ inv_ez @ sp.hstack([dx0, dy0]) @ exy / beta +
          beta * foo) / omega
  h2e = inv_exy @ (sp.vstack([dy0, -dx0]) @ sp.hstack([dx1, dy1]) / beta +
                   beta * foo) / omega
  return e2h, h2e


def _power_in_mode(emode, beta, omega, epsilon, dx, dy):
  """Compute poynting vector over `emode`."""
  e2h, _ = _conversion_operators(beta, omega, epsilon, dx, dy)
  hmode = np.reshape(e2h @ np.ravel(emode), emode.shape)
  return np.sum(emode[0] * hmode[1] * (dx[:, 1])[:, None] * dy[:, 0] -
                emode[1] * hmode[0] * (dx[:, 0])[:, None] * dy[:, 1])


def waveguide(i, omega, epsilon, dx, dy):
  """Solves for the `i`th mode of the waveguide at `omega`.

  Assumes a real-valued structure in the x-y plane and propagation along the
  z-axis according to `exp(-i * wavevector* z)`. Uses dimensionless units and
  periodic boundaries.

  Currently does not use JAX and is not differentiable -- waiting on a
  sparse eigenvalue solver in JAX.

  Args:
    i: Mode number to solve for, where `i = 0` corresponds to the fundamental
      mode of the structure.
    omega: Angular frequency of the mode.
    epsilon: `(3, xx, yy)` array of permittivity values for Ex, Ey, and Ez
      nodes on a finite-difference Yee grid.
    dx: `(xx, 2)` array of cell sizes in the x-direction. `[:, 0]` values
      correspond to Ey/Ez components while `[:, 1]` values correspond to
      Ex components.
    dy: `(yy, 2)` array similar to `dx` but for cell sizes along `y`.

  Returns:
    wavevector: Real-valued scalar.
    fields: `(2, xx, yy)` array of real-valued Ex and Ey field values of the
      mode. Normalized such that the overlap integral with the output field
      squared is equal to the power in the mode.

  """
  A = _waveguide_operator(omega, epsilon, dx, dy)
  shift = _find_largest_eigenvalue(A, 20)
  if shift >= 0:
    raise ValueError("Expected largest eigenvalue to be negative.")
  w, v = sp.linalg.eigs(A - shift * sp.eye(A.shape[0]), k=i+1, which="LM")
  beta = np.real(np.sqrt(w[i] + shift))
  mode = np.reshape(np.real(v[:, i]), (2,) + epsilon.shape[1:])
  if beta == 0:
    raise ValueError("No propagating mode found.")
  mode /= np.linalg.norm(np.ravel(mode), ord=2)
  mode /= np.sqrt(_power_in_mode(mode, beta, omega, epsilon, dx, dy))
  return np.float32(beta), np.float32(mode)
