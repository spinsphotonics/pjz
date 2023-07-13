"""Define inputs and outputs."""

import jax

from typing import Any, List, Optional, Tuple


def _forward_difference(arr, axis):
  return jnp.roll(arr, axis) - arr


def _dxf(dx, dy):
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


# TODO: Test this with batch dimension?
def wg_port(
        epsilon: jax.Array,
        omega: jax.Array,
        position: Tuple[Any, Any, Any],
        width: Tuple[int, int, int],
        num_modes: int,
        shift_iters: int = 10,
        max_iters: int = 1000000,
        tol: float = 1e-6,
) -> List[Tuple[jax.Array, jax.Array, Tuple[int, int, int]]]:
  """Waveguide modes.

  Try to do something so that the excitation is overall positive (what we
  really want is to have a non-random phase to the excitation values).

  Args:
    epsilon: Array of current permittivity values with
      ``epsilon.shape[-4:] == (3, xx, yy, zz)``.
    omega: Array of angular frequencies at which to solve for waveguide modes.
    position: ``(x0, y0, z0)`` position for computing modes within ``epsilon``.
    width: ``(xx, yy, zz)`` number of cells in which to form a window to compute
      mode fields. Mode computation will occur within cell indices within
      ``(x0, y0, z0)`` to ``(x0 + xx - 1, y0 + yy - 1, z0 + zz - 1)`` inclusive.
    num_modes: Positive integer denoting the number of modes to solve for.
    shift_iters: Number of iterations used to determine the largest eigenvalue
      of the waveguide operator.
    num_subspace_iters: Maximum number of eigenvalue solver iterations to execute.
    tol: Error threshold for eigenvalue solver.

  Returns:
    ``num_modes`` tuples of form ``(excitation, wavevector, position)``, in
    order of decreasing ``wavevector`` so that the fundamental mode is at
    index ``0``.

  """
  pass


def wg_port_hot(
        epsilon: jax.Array,
        omega: jax.Array,
        hot_start: List[Tuple[jax.Array, jax.Array, Tuple[int, int, int]]],
        shift_iters: int = 10,
        max_iters: int = 1000000,
        tol: float = 1e-6,
) -> List[Tuple[jax.Array, jax.Array, Tuple[int, int, int]]]:
  """Same as ``wg_port()`` but with non-random initial value.

  Say something about this being quicker...
  Also useful to make sure we don't keep on flipping signs.

  Args:
    epsilon: Same as ``wg_port()``.
    omega: Same as ``wg_port()``.
    hot_start: Use the output from a previous ``wg_port()`` call as an initial
      starting point.
    shift_iters: Same as ``wg_port()``. 
    max_iters: Same as ``wg_port()``. 
    tol: Same as ``wg_port()``. 

  Returns:
    Same as ``wg_port()``.

  """
  pass
