"""Waveguide modes."""

from functools import partial
from typing import Optional, Tuple
from warnings import warn

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
    omega: Array with shape ``(ww,)``.
    shift: Array with shape ``(ww,)``.

  Returns:
    Function applying operator to input with shape ``(2, xx, yy, num_modes)``.
  """
  eps_yx = epsilon[None, (1, 0), :, :, None]
  eps_z = epsilon[None, 2, :, :, None]
  omega = omega[:, None, None, None, None]
  shift = shift[:, None, None, None, None]

  def _apply(arr):
    """Apply the operator to array with shape ``(ww, 2, xx, yy, num_modes)``."""
    # jax.debug.print("{o}, {e}, {s}, {a}", o=omega.shape,
    #                 e=eps_yx.shape, s=shift.shape, a=arr.shape)
    a = (omega**2 * eps_yx - shift) * arr
    b = -_diff(arr[:, 0], -2, False) + _diff(arr[:, 1], -3, False)
    b /= eps_z
    b = jnp.stack([-_diff(b, -2, True), _diff(b, -3, True)], axis=1)
    b *= eps_yx
    c = _diff(arr[:, 0], -3, True) + _diff(arr[:, 1], -2, True)
    c = jnp.stack([_diff(c, -3, False), _diff(c, -2, False)], axis=1)
    # jax.debug.print("{a}, {b}, {c}", a=a.shape, b=b.shape, c=c.shape)
    return a + b + c
  return _apply


def _cross(arr, dx, dy, dz):
  fx, fy, fz = arr[:, 0], arr[:, 1], arr[:, 2]
  # jax.debug.print("{x} {y} {z}, {x1} {y1} {z1}", x=fx.shape, y=fy.shape,
  #                 z=fz.shape, x1=dx(fz).shape, y1=dy(fx).shape, z1=dz(fx).shape)
  return jnp.stack([dy(fz) - dz(fy), dz(fx) - dx(fz), dx(fy) - dy(fx)], axis=1)


def _curl(beta, arr, is_forward):
  """Apply the curl operator to field of shape ``(ww, 3, xx, yy, num_modes)``."""
  return _cross(
      arr,
      dx=partial(_diff, axis=-3, is_forward=is_forward),
      dy=partial(_diff, axis=-2, is_forward=is_forward),
      dz=lambda x: -1j * beta[:, None, None, :] * x,
  )


def _fullh(beta, x):
  """Get the full H-field."""
  dx = partial(_diff, axis=-3, is_forward=True)
  dy = partial(_diff, axis=-2, is_forward=True)
  return jnp.stack([x[:, 0],
                    x[:, 1],
                    (dx(x[:, 0]) + dy(x[:, 1])) / (1j * beta[:, None, None, :])],
                   axis=1)


def _poynting(beta, omega, epsilon, x):
  h = _fullh(beta, x)
  e = _curl(beta, h, is_forward=False) / \
      (1j * omega[:, None, None, None, None] * epsilon[None, ..., None])
  return jnp.real(jnp.sum(e[:, 0] * h[:, 1] - e[:, 1] * h[:, 0], axis=(1, 2)))


def _full_fields(beta, omega, epsilon, x):
  # jax.debug.print("shapes {b} {x}", b=beta.shape, x=x.shape)
  h = _fullh(beta, x)
  # jax.debug.print("h shape {h}", h=h.shape)
  e = _curl(beta, h, is_forward=False) / (
      1j * omega[:, None, None, None, None] * epsilon[None, ..., None])
  # jax.debug.print("e shape {e}", e=e.shape)
  h2 = _curl(beta, e, is_forward=True) / \
      (-1j * omega[:, None, None, None, None])

  return h, e, h2


def _subspace_iteration(op, x, n, tol):
  """Perform the subspace iteration.

  Args:
    op: ``op(x)`` applies the operator to ``x``.
    x: Array with shape ``(ww, 2, xx, yy, mm)``.
    n: Maximum number of iterations.
    tol: Error threshold.

  Returns:
    ``(w, x, iters, err)`` with ``w`` and ``err`` having shape ``(ww, mm)``.
  """
  # jax.debug.print("starting _subspace_iteration with {x}", x=x.shape)

  # Uses the format
  # ``(eigenvalues, eigenvectors, op(eigenvectors), iteration_number, errors)``.
  ww, mm = x.shape[0], x.shape[-1]
  vals = (jnp.zeros((ww, mm)), x, op(x),
          jnp.array(0), jnp.inf * jnp.ones((ww, mm)))

  def cond(vals):
    _, _, _, i, err = vals
    return jnp.logical_and(i < n, jnp.max(err) > tol)

  def orthogonalize(x):
    y, _ = jnp.linalg.qr(jnp.reshape(x, (ww, -1, mm)), mode="reduced")
    return jnp.reshape(y, x.shape)

  def body(vals):
    w, x, y, i, err = vals
    x = orthogonalize(y)
    y = op(x)
    w = jnp.sum(x * y, axis=(1, 2, 3))
    err = jnp.linalg.norm(jnp.reshape(
        y - w[:, None, None, None, :] * x, (ww, -1, mm)), axis=1)
    return w, x, y, i + 1, err

  w, x, _, iters, err = jax.lax.while_loop(cond, body, vals)

  # Extract eigenvalues and re-order.
  inds = jnp.flip(jnp.argsort(w, axis=-1), axis=-1)
  x = jnp.take_along_axis(x, inds[:, None, None, None, :], axis=-1)
  w = jnp.take_along_axis(w, inds, axis=-1)
  err = jnp.take_along_axis(err, inds, axis=-1)
  return w, x, err, iters


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

  Uses the subspace iteration method to obtain modes without leaving JAX.
  Allows for updating mode solutions via the ``init`` parameter (e.g. for 
  small changes in ``epsilon`` and/or ``omega``).

  Args:
    epsilon: ``(3, xx, yy, zz)`` array of permittivity values with exactly one
      ``xx``, ``yy``, or ``zz`` equal to ``1``.
    omega: ``(ww,)`` array of real-valued scalar angular frequencies.
    num_modes: Integer denoting number of modes to solve for.
    init: ``(ww, 2, xx, yy, zz, num_modes)`` of values to use as initial guess.
    shift_iters: Number of iterations used to determine the largest eigenvalue
      of the waveguide operator.
    max_iters: Maximum number of eigenvalue solver iterations to execute.
    tol: Error threshold for eigenvalue solver.

  Returns:
    ``(wavevector, excitation, err, iters)`` where 
    ``iters`` is the number of executed solver iterations,
    ``excitation.shape == (ww, 2, xx, yy, zz, num_modes)`` and
    ``wavevector.shape == err.shape == (ww, num_modes)``, with 
    ``excitation[..., i]``, ``wavevector[i]``, and ``err[i]``  being ordered
    such that ``i == 0`` corresponds to the fundamental mode.

  """
  if type(omega) == float or omega.shape == ():
    omega = jnp.array([omega])
    warn("mode() expects the ``omega`` parameter to have shape ``(ww,)``, "
         "but got scalar instead.")

  if not (1 in epsilon.shape):
    raise ValueError(
        f"Expected exactly one of the spatial dimensions of ``epsilon`` to be "
        f"singular, instead got ``epsilon.shape == {epsilon.shape}``.")
  prop_axis = "xyz"[epsilon.shape.index(1) - 1]

  mode_shape = ((omega.shape[0], 2) +
                jnp.squeeze(epsilon).shape[1:] + (num_modes,))

  if init is None:
    init = _random(mode_shape)
  else:
    init = jnp.squeeze(init, "xyz".index(prop_axis) + 2)

    # Convert to output form.
    if prop_axis == "y":
      init = jnp.flip(jnp.swapaxes(init, 2, 3), axis=(1, 2))
    elif prop_axis == "z":
      init *= jnp.array([1, -1])[None, :, None, None, None]

    init = jnp.flip(init, axis=1)  # Excitation-to-field flip.

  # Prepare inputs into "propagate-along-z" form.
  if prop_axis == "x":
    epsilon = epsilon[(1, 2, 0), ...]
  elif prop_axis == "y":
    epsilon = jnp.flip(jnp.swapaxes(epsilon[(2, 0, 1), ...], 1, 3), axis=1)
    mode_shape = tuple(mode_shape[i] for i in (0, 1, 3, 2, 4))

  epsilon = jnp.squeeze(epsilon)
  init = jnp.reshape(init, mode_shape)

  # Get largest eigenvalue to use as shift.
  shift, _, _, _ = _subspace_iteration(
      _operator(epsilon, omega, jnp.zeros_like(omega)),
      _random(mode_shape[:-1] + (1,)),
      shift_iters,
      tol,
  )

  # Solve for modes.
  w, x, err, iters = _subspace_iteration(
      _operator(epsilon, omega, shift[:, 0]),
      init,
      max_iters,
      tol,
  )

  wavevector = jnp.sqrt(w + shift)

  # Normalize the modes.
  x /= jnp.sqrt(_poynting(wavevector, omega, epsilon, x)
                )[:, None, None, None, :]
  # jax.debug.print("x shape {x}", x=x.shape)

  exc = jnp.flip(x, axis=1)  # Field-to-excitation flip.

  # Convert to output form.
  if prop_axis == "y":
    exc = jnp.swapaxes(jnp.flip(exc, axis=(1, 2)), 2, 3)
    # jax.debug.print("y exc shape {exc}", exc=exc.shape)
  elif prop_axis == "z":
    exc *= jnp.array([1, -1])[None, :, None, None, None]
    # jax.debug.print("z exc shape {exc}", exc=exc.shape)
  # jax.debug.print("exc shape {exc}", exc=exc.shape)
  exc = jnp.expand_dims(exc, "xyz".index(prop_axis) + 2)

  # jax.debug.print("exc shape {exc}", exc=exc.shape)

  return wavevector, exc, err, iters
