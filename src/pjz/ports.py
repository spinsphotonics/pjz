"""Define inputs and outputs."""

from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp


def _diff(arr, axis, is_forward):
  if is_forward:
    return jnp.roll(arr, -1, axis) - arr
  else:
    return arr - jnp.roll(arr, 1, axis)


def _operator(epsilon, omega, shift):
  """Returns the waveguide operator.
   
  Args:
    epsilon: Array with shape ``(..., 3, xx, yy)``.
    omega: Array.
    shift: Array.
  """
  eps_yx = epsilon[..., (1, 0), :, :, None]
  eps_z = epsilon[..., 2, :, :, None]
  omega = omega[..., None, None, None, None]
  shift = shift[..., None, None, None, None]

  def _apply(v):
    arr = jnp.reshape(v,
                      v.shape[:-2] + (2,) + epsilon.shape[-2:] + (v.shape[-1],))
    a = (omega**2 * eps_yx - shift) * arr
    b = (-_diff(arr[..., 0, :, :, :], -2, False) +
         _diff(arr[..., 1, :, :, :], -3, False))
    b /= eps_z
    b = jnp.stack([-_diff(b, -2, True), _diff(b, -3, True)], axis=-4)
    b *= eps_yx
    c = (_diff(arr[..., 0, :, :, :], -3, True) +
         _diff(arr[..., 1, :, :, :], -2, True))
    c = jnp.stack([_diff(c, -3, False), _diff(c, -2, False)], axis=-4)
    return jnp.reshape(a + b + c, v.shape)
  return _apply


def _subspace_iteration(op, x, n, tol):
  """Perform the subspace iteration.

  Args:
    op: ``op(x)`` applies the operator to ``x``.
    x: ``x.shape[-2:] == (n, k)``.
    n: Maximum number of iterations.
    tol: Error threshold.

  Returns:
    ``(w, x, iters, err)`` with ``w.shape[-1] == err.shape[-1] == k``.

  """

  def cond(vals):
    _, _, _, i, err = vals
    return jnp.logical_and(i < n, jnp.max(err) > tol)

  def body(vals):
    w, x, y, i, err = vals
    x, _ = jnp.linalg.qr(y, mode="reduced")
    y = op(x)
    w = jnp.sum(x * y, axis=-2)
    err = jnp.linalg.norm(y - w[..., None, :] * x, axis=-2)
    return w, x, y, i + 1, err

  init_shape = x.shape[:-2] + (x.shape[-1],)
  init_vals = (
      jnp.zeros(init_shape),
      x,
      op(x),
      jnp.array(0),
      jnp.inf * jnp.ones(init_shape),
  )
  w, x, _, iters, err = jax.lax.while_loop(cond, body, init_vals)

  # Extract eigenvalues and re-order.
  inds = jnp.flip(jnp.argsort(w, axis=-1), axis=-1)
  w = jnp.take_along_axis(w, inds, axis=-1)
  x = jnp.take_along_axis(x, inds[..., None, :], axis=-1)
  err = jnp.take_along_axis(err, inds, axis=-1)
  return w, x, err, iters


def _slice_axis(width):
  if width[2] == 1:
    return "z"
  elif width[1] == 1:
    return "y"
  elif width[0] == 1:
    return "x"
  else:
    raise ValueError("``width`` must have at least one singular dimension.")


def _extract_slice(epsilon, p, w, slice_axis):
  if slice_axis == "z":
    return epsilon[..., :, p[0]:p[0] + w[0], p[1]:p[1] + w[1], p[2]]

  elif slice_axis == "y":
    return jnp.swapaxes(
        epsilon[..., (2, 0, 1), p[0]:p[0] + w[0], p[1], p[2]:p[2] + w[2]],
        axis1=-2,
        axis2=-1,
    )

  elif slice_axis == "x":
    return epsilon[..., (1, 2, 0), p[0], p[1]:p[1] + w[1], p[2]:p[2] + w[2]]

  else:
    raise ValueError("Unrecognized ``slice_axis``")


def _excitation_form(x, width, slice_axis):
  if slice_axis == "z":
    x = jnp.reshape(x, x.shape[:-2] + (2,) + width + (x.shape[-1],))

  elif slice_axis == "y":
    print(x.shape)
    x = jnp.reshape(x, x.shape[:-2] + (2,) + width[::-1] + (x.shape[-1],))
    x = jnp.swapaxes(x[..., (1, 0), :, :, :, :], -4, -2)

  elif slice_axis == "x":
    x = jnp.reshape(x, x.shape[:-2] + (2,) + width + (x.shape[-1],))

  else:
    raise ValueError("Unrecognized ``slice_axis``")

  # Swap components for excitation.
  return x[..., (1, 0), :, :, :, :]


def _vector_form(excitation, slice_axis):
  """Opposite of ``_excitation_form()``."""
  x = excitation[..., (1, 0), :, :, :, :]

  if slice_axis == "y":
    x = jnp.swapaxes(x[..., (1, 0), :, :, :, :], -4, -2)

  return jnp.reshape(x, x.shape[:-5] + (-1,) + x.shape[-1:])


def _random(epsilon, omega, width, k):
  batch_dims = jnp.broadcast_shapes(epsilon.shape[:-4], omega.shape)
  n = 2 * width[0] * width[1] * width[2]
  return jax.random.uniform(jax.random.PRNGKey(0), batch_dims + (n, k))


def wg_port(
        epsilon: jax.Array,
        omega: jax.Array,
        position: Tuple[int, int, int],
        width: Tuple[int, int, int],
        num_modes: int,
        shift_iters: int = 10,
        max_iters: int = 100000,
        tol: float = 1e-4,
):
  """Waveguide modes.

  Try to do something so that the excitation is overall positive (what we
  really want is to have a non-random phase to the excitation values).

  Args:
    epsilon: Array of current permittivity values with
      ``epsilon.shape[-4:] == (3, xx, yy, zz)``.
    omega: Array of angular frequencies at which to solve for waveguide modes.
    position: ``(x0, y0, z0)`` position for computing modes within ``epsilon``.
    width: ``(wx, wy, wz)`` number of cells in which to form a window to compute
      mode fields. Mode computation will occur within cell indices within
      ``(x0, y0, z0)`` to ``(x0 + xx - 1, y0 + yy - 1, z0 + zz - 1)`` inclusive.
    num_modes: Positive integer denoting the number of modes to solve for.
    shift_iters: Number of iterations used to determine the largest eigenvalue
      of the waveguide operator.
    max_iters: Maximum number of eigenvalue solver iterations to execute.
    tol: Error threshold for eigenvalue solver.

  Returns:
    ``(excitation, wavevector, error, iters)`` where
    ``excitation.shape[-4:] == (2, wx, wy, wz, num_modes)``,
    ``wavevector.shape[-1] == num_modes``, 
    ``error.shape[-1] == num_modes``, and ``iters`` is an integer denoting the
    number of iterations executed.
    With respect to the last axis of the ``wavevector``, ``excitation``, and
    ``error`` outputs, these are ordered by decreasing ``wavevector`` so that
    the fundamental mode is at index ``0``.

  """
  # print(excitation.shape)
  x = _random(epsilon, omega, width, num_modes)
  print(x.shape)
  excitation = _excitation_form(x, width, _slice_axis(width))
  return wg_port_hot(epsilon, omega, position, excitation, shift_iters, max_iters, tol)


def wg_port_hot(
        epsilon: jax.Array,
        omega: jax.Array,
        position: Tuple[int, int, int],
        excitation: jax.Array,
        shift_iters: int = 10,
        max_iters: int = 100000,
        tol: float = 1e-4,
) -> List[Tuple[jax.Array, jax.Array, Tuple[int, int, int]]]:
  """Same as ``wg_port()`` but with non-random initial value.

  Say something about this being quicker...
  Also useful to make sure we don't keep on flipping signs.

  Args:
    epsilon: Same as ``wg_port()``.
    omega: Same as ``wg_port()``.
    position: Same as ``wg_port()``.
    excitation: Use the ``excitation`` output from a previous ``wg_port()`` call
      as an initial starting point.
    shift_iters: Same as ``wg_port()``. 
    max_iters: Same as ``wg_port()``. 
    tol: Same as ``wg_port()``. 

  Returns:
    Same as ``wg_port()``.

  """
  width = excitation.shape[-4:-1]
  slice_axis = _slice_axis(width)
  epsilon = _extract_slice(epsilon, position, width, slice_axis)

  # Get largest eigenvalue to use as shift.
  shift, _, _, _ = _subspace_iteration(
      _operator(epsilon, omega, jnp.zeros_like(omega)),
      _random(epsilon, omega, width, 1),
      shift_iters,
      tol,
  )

  # Solve for modes.
  w, x, err, iters = _subspace_iteration(
      _operator(epsilon, omega, shift[0]),
      _vector_form(excitation, slice_axis),
      max_iters,
      tol,
  )

  # Push into ports format.
  wavevector = jnp.sqrt(w + shift)
  excitation = _excitation_form(x, width, slice_axis)

  return excitation, wavevector, err, iters
