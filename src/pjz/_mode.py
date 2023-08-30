"""Waveguide modes."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def _random(shape):
  return jax.random.normal(jax.random.PRNGKey(0), shape)


def _diff(arr, axis, is_forward):
  if is_forward:
    return jnp.roll(arr, -1, axis) - arr
  else:
    return arr - jnp.roll(arr, 1, axis)


def _operator(epsilon, omega, shift):
  """Returns the waveguide operator.
   
  Args:
    epsilon: Array with shape ``(3, xx, yy)``.
    omega: Array with shape ``()``.
    shift: Array with shape ``()``.

  Returns:
    Function applying operator to input with shape ``(2, xx, yy, num_modes)``.
  """
  eps_yx = epsilon[(1, 0), :, :, None]
  eps_z = epsilon[2, :, :, None]

  def _apply(arr):
    """Apply the operator to array with shape ``(2, xx, yy, num_modes)``."""
    a = (omega**2 * eps_yx - shift) * arr
    b = -_diff(arr[0], -2, False) + _diff(arr[1], -3, False)
    b /= eps_z
    b = jnp.stack([-_diff(b, -2, True), _diff(b, -3, True)])
    b *= eps_yx
    c = _diff(arr[0], -3, True) + _diff(arr[1], -2, True)
    c = jnp.stack([_diff(c, -3, False), _diff(c, -2, False)])
    return a + b + c
  return _apply


def _subspace_iteration(op, x, n, tol):
  """Perform the subspace iteration.

  Args:
    op: ``op(x)`` applies the operator to ``x``.
    x: Array with shape ``(2, xx, yy, num_modes)``.
    n: Maximum number of iterations.
    tol: Error threshold.

  Returns:
    ``(w, x, iters, err)`` with ``w.shape[-1] == err.shape[-1] == k``.

  """

  # Uses the format
  # ``(eigenvalues, eigenvectors, op(eigenvectors), iteration_number, errors)``.
  k = x.shape[-1]
  vals = (jnp.zeros(k), x, op(x), jnp.array(0), jnp.inf * jnp.ones(k))

  def cond(vals):
    _, _, _, i, err = vals
    return jnp.logical_and(i < n, jnp.max(err) > tol)

  def orthogonalize(x):
    y, _ = jnp.linalg.qr(jnp.reshape(x, (-1, x.shape[-1])), mode="reduced")
    return jnp.reshape(y, x.shape)

  def body(vals):
    w, x, y, i, err = vals
    x = orthogonalize(y)
    y = op(x)
    w = jnp.sum(x * y, axis=(0, 1, 2))
    err = jnp.linalg.norm(jnp.reshape(y - w * x, (-1, k)), axis=0)
    return w, x, y, i + 1, err

  w, x, _, iters, err = jax.lax.while_loop(cond, body, vals)

  # Extract eigenvalues and re-order.
  inds = jnp.flip(jnp.argsort(w))
  x = jnp.take_along_axis(x, inds[None, None, None, :], axis=-1)
  return w[inds], x, err[inds], iters


def mode(
        epsilon: jax.Array,
        omega: jax.Array,
        num_modes: int,
        init: Optional[jax.Array] = None,
        shift_iters: int = 10,
        max_iters: int = 100000,
        tol: float = 1e-4,
) -> Tuple[jax.Array, jax.Array, jax.Array, int]:
  """Solve for waveguide modes.

  Args:
    epsilon: ``(3, xx, yy, zz)`` array of permittivity values with exactly one
      ``xx``, ``yy``, or ``zz`` equal to ``1``.
    omega: Real-valued scalar angular frequency.
    num_modes: Integer denoting number of modes to solve for.
    init: ``(2, xx, yy, zz, num_modes)`` of values to use as initial guess.
    shift_iters: Number of iterations used to determine the largest eigenvalue
      of the waveguide operator.
    max_iters: Maximum number of eigenvalue solver iterations to execute.
    tol: Error threshold for eigenvalue solver.

  Returns:
    ``(wavevector, excitation, err, iters)`` where 
    ``iters`` is the number of executed solver iterations, and
    ``excitation.shape == (2, xx, yy, zz, num_modes)`` and
    ``wavevector.shape == err.shape == (num_modes,)``, with 
    ``excitation[..., i]``, ``wavevector[i]``, and ``err[i]``  being ordered
    such that ``i == 0`` corresponds to the fundamental mode.

  """
  if not (1 in epsilon.shape):
    raise ValueError(
        f"Expected exactly one of the spatial dimensions of ``epsilon`` to be "
        f"singular, instead got ``epsilon.shape == {epsilon.shape}``.")
  prop_axis = "xyz"[epsilon.shape.index(1) - 1]

  if init is None:
    init = _random((2,) + epsilon.shape[1:] + (num_modes,))

  # Prepare inputs into "propagate-along-z" form.
  if prop_axis == "x":
    epsilon = epsilon[(1, 2, 0), ...]
  elif prop_axis == "y":
    # TODO: Need to fix this...
    epsilon = jnp.flip(jnp.swapaxes(epsilon[(2, 0, 1), ...], 1, 3), axis=1)
    # TODO: May need to scale one of the components by `-1`.
    init = jnp.flip(jnp.swapaxes(init[(1, 0), ...], 1, 3), axis=1)

  epsilon = jnp.squeeze(epsilon)
  init = jnp.reshape(init, (2,) + epsilon.shape[1:] + (num_modes,))

  # Get largest eigenvalue to use as shift.
  shift, _, _, _ = _subspace_iteration(
      _operator(epsilon, omega, jnp.zeros_like(omega)),
      _random((2,) + epsilon.shape[1:] + (1,)),
      shift_iters,
      tol,
  )

  # Solve for modes.
  w, x, err, iters = _subspace_iteration(
      _operator(epsilon, omega, shift),
      jnp.flip(init, axis=0),  # Excitation-to-field flip.
      max_iters,
      tol,
  )
  x = jnp.flip(x, axis=0)  # Field-to-excitation flip.

  # Convert to output form.
  wavevector = jnp.sqrt(w + shift)
  if prop_axis == "y":
    x = jnp.swapaxes(jnp.flip(x[(1, 0), ...], axis=1), 1, 2)
  if prop_axis == "z":
    x *= jnp.array([1, -1])[:, None, None, None]
  excitation = jnp.expand_dims(x, "xyz".index(prop_axis) + 1)

  return wavevector, excitation, err, iters
