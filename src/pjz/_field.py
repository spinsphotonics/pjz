"""Time-harmonic fields and scattering parameters."""

from functools import partial
from typing import Any, Dict, NamedTuple, Sequence, Tuple

import fdtdz_jax
import jax
import jax.numpy as jnp
import numpy as np


class SimParams(NamedTuple):
  """Simulation parameters for interfacing with the ``fdtdz`` package.

  Parameterize ``fdtdz.fdtdz()`` for the specific case of extracting
  time-harmonic solutions.

  Attributes:
    omega_range: ``(omega_min, omega_max)`` range of possible values for the
      angular frequencies to be extracted from a single simulation.
    tt: Number of FDTD updates to perform.
    source_ramp: Number of periods over which to bring sources to steady state.
    source_delay: Number of periods before starting the source ramp.
    absorption_padding: Number of cells additional padding cells in the x-y
      plane to use for adiabatic absorbing conditions.
    absorption_coeff: Strength of the adiabatic absorber.
    pml_widths: ``(pml_lo, pml_hi)`` denoting the number of PML cells to use
      at the bottom and top of the simulation domain. See ``fdtdz.fdtdz()``
      docstring for additional details.
    pml_alpha_coeff: Strength of ``alpha`` parameter of the PML, see the
      ``fdtdz.fdtdz()`` docstring for more details.
    pml_sigma_lnr: Controls the ``sigma`` parameter of the PML, see the
      ``fdtdz.fdtdz()`` docstring for more details.
    pml_sigma_m: Controls the ``sigma`` parameter of the PML, see the
      ``fdtdz.fdtdz()`` docstring for more details.
    use_reduced_precision: See the ``fdtdz.fdtdz()`` docstring for more details.
    launch_params: See the ``fdtdz.fdtdz()`` docstring for more details.

  """
  omega_range: Tuple[float, float]
  tt: int
  dt: float = 0.5
  source_ramp: float = 4.0
  source_delay: float = 4.0
  absorption_padding: int = 50
  absorption_coeff: float = 1e-4
  pml_widths: Tuple[int, int] = (16, 16)
  pml_alpha_coeff: float = 0.0
  pml_sigma_lnr: float = 0.5
  pml_sigma_m: float = 1.3
  use_reduced_precision: bool = True
  launch_params: Any = None


def _zz(pml_widths, use_reduced_precision):
  """Height of fdtdz simulation domain."""
  return (128 if use_reduced_precision else 64) - sum(pml_widths)


def _pad_zz(epsilon_zz, pml_widths, use_reduced_precision):
  """Amount of padding needed in z-direction to fill-out fdtdz domain."""
  zz = _zz(pml_widths, use_reduced_precision)
  bot = (zz - epsilon_zz) // 2
  top = zz - epsilon_zz - bot
  return (bot, top)


def _absorption_profiles(numcells, width, smoothness):
  """1D quadratic profile for adiabatic absorption boundary conditions."""
  center = (numcells - 1) / 2
  offset = jnp.array([[0], [0.5]])
  pos = jnp.arange(numcells) + offset
  pos = jnp.abs(pos - center) - center + width
  pos = jnp.clip(pos, a_min=0, a_max=None)
  return smoothness * jnp.power(pos, 2)


def _cross_profiles(x, y):
  """Combine two 1D absorption profiles into a 2D profile."""
  return jnp.maximum(*jnp.meshgrid(x, y, indexing='ij'))[None, ...]


def _absorption_mask(xx, yy, width, smoothness):
  """`(3, xx, yy)` absorption values in the x-y plane."""
  x = _absorption_profiles(xx, width, smoothness)
  y = _absorption_profiles(yy, width, smoothness)
  return jnp.concatenate([_cross_profiles(x[0], y[1]),
                          _cross_profiles(x[1], y[0]),
                          _cross_profiles(x[1], y[1])])


def _safe_div(x, y):
  """Division where divide-by-zero yields ``0``."""
  return jnp.zeros_like(x) if y == 0 else x / y


def _pml_sigma(pml_widths, zz, ln_R, m):
  """`(zz, 2)` array of conductivity values for PML along the z-axis."""
  offset = jnp.array([[0], [0.5]])
  z = jnp.arange(zz) + offset
  z = jnp.stack([_safe_div(pml_widths[0] - z, pml_widths[0]),
                 _safe_div(z + 0.5 - zz + pml_widths[1], pml_widths[1])],
                axis=-1)
  z = jnp.max(jnp.clip(z, a_min=0, a_max=None), axis=-1)
  return ((m + 1) * ln_R * z**m).T


def _ramped_sin(omega, width, delay, dt, tt):
  """Sine function with a gradual ramp."""
  t = omega[:, None] * dt * jnp.arange(tt)
  return jnp.mean(((1 + jnp.tanh(t / width - delay)) / 2) * jnp.sin(t), axis=0)


def _sampling_interval(
        omega_min: float,
        omega_max: float,
        omega_n: int,
        dt: float
) -> int:
  """Sampling interval to observe `n` components within `[wmin, wmax]`."""
  # Period of average angular frequency.
  period = 4 * np.pi / (omega_max + omega_min) / dt

  if omega_n == 1:
    # If only one frequency is needed then we just sample at a quarter wavelength.
    return int(round(period / 4))
  else:
    # Otherwise, we find the interval at which the maximum and minimum angular
    # frequencies generate a phase difference of `pi * (n - 1) / n`.
    cutoff = np.pi * (omega_n - 1) / omega_n / (omega_max - omega_min) / dt

    # Now, we use the the periodicity of the center frequency to find the nearest
    # "open window" (a time step in which no frequency component will complete a
    # rotation which is a multiple of `pi` -- since this would make this component
    # unobservable), and use this as our sampling interval.
    m = np.floor((cutoff - period / 4) / (period / 2))
    return int(round(period / 4 + m * period / 2))


def _output_phases(
    omega: jax.Array,
    output_steps: Tuple[int, ...],
    dt: float,
) -> jax.Array:
  """Output phases for angular frequencies ``omega`` at ``output_steps``."""
  steps = jnp.arange(*output_steps)
  theta = omega[:, None] * dt * steps
  return jnp.concatenate([jnp.cos(theta), -jnp.sin(theta)], axis=0)


# TODO: Use jax arrays for source_pos so we don't need to mark it as static.
@partial(jax.jit, static_argnames=["source_pos", "sim_params"])
def field(
    epsilon: jax.Array,
    source: jax.Array,
    omega: jax.Array,
    source_pos: int,
    sim_params: SimParams,
):
  """Time-harmonic solution of Maxwell's equations.

  Args:
    epsilon: ``(3, xx, yy, zz)`` array of permittivity values.
    source: Array of excitation values of shape ``(2, 1, yy, zz)``,
      ``(2, xx, 1, yy)``, or ``(2, xx, yy, 1)``.
    omega: ``(ww,)`` array of angular frequencies.
    source_pos: Position of source along axis of propagation.
    sim_params: Simulation parameters.

  Returns:
    ``(ww, 3, xx, yy, zz)`` array of complex-valued field values at the various
    ``omega``.
  """
  # TODO: Consider doing some shape testing
  (omega_range, tt, dt, source_ramp, source_delay, absorption_padding,
   absorption_coeff, pml_widths, pml_alpha_coeff, pml_sigma_lnr, pml_sigma_m,
   use_reduced_precision, launch_params) = sim_params

  # TODO: Document this.
  pad_zz = _pad_zz(epsilon.shape[3], pml_widths, use_reduced_precision)
  padding = ((absorption_padding, absorption_padding),
             (absorption_padding, absorption_padding),
             pad_zz)

  # Take care of input waveform and output transform.
  source_waveform = _ramped_sin(omega, source_ramp, source_delay, dt, tt)
  source_waveform = jnp.pad(source_waveform[:, None], ((0, 0), (0, 1)))

  interval = _sampling_interval(
      omega_range[0], omega_range[1], omega.shape[0], dt)
  output_steps = (tt - 2 * interval * omega.shape[0] - 1, tt, interval)

  # Manipulate source into the form that ``fdtdz_jax.fdtdz()`` expects.
  for i in range(3):
    if source.shape[i + 1] == 1:
      source_pos += padding[i][0]
    else:
      source_padding = (i + 1) * ((0, 0),) + \
          (padding[i],) + (2 - i) * ((0, 0),)
      source = jnp.pad(source, source_padding)

  if (source.shape[1] == 1 or source.shape[2] == 1) and source_pos % 2 == 1:
    # Only even values of source position are allowed for x and y propagation.
    source_pos += 1
    source_waveform = jnp.flip(source_waveform, axis=1)
  elif source.shape[3] == 1:
    source = jnp.pad(source[None, ...], ((0, 1),) + 4 * ((0, 0),))

  # Boundary conditions.
  absorption_mask = _absorption_mask(epsilon.shape[1] + 2 * absorption_padding,
                                     epsilon.shape[2] + 2 * absorption_padding,
                                     absorption_padding, absorption_coeff)

  pml_sigma = _pml_sigma(pml_widths, _zz(pml_widths, use_reduced_precision),
                         pml_sigma_lnr, pml_sigma_m)
  pml_kappa = jnp.ones_like(pml_sigma)
  pml_alpha = pml_alpha_coeff * jnp.ones_like(pml_sigma)

  # Simulate.
  fields = fdtdz_jax.fdtdz(
      epsilon=epsilon,
      dt=dt,
      source_field=source,
      source_waveform=source_waveform,
      source_position=source_pos,
      absorption_mask=absorption_mask,
      pml_kappa=pml_kappa,
      pml_sigma=pml_sigma,
      pml_alpha=pml_alpha,
      pml_widths=pml_widths,
      output_steps=output_steps,
      use_reduced_precision=use_reduced_precision,
      launch_params=launch_params,
      offset=(padding[0][0], padding[1][0], padding[2][0]),
  )

  # Convert to complex.
  output_phases = _output_phases(omega, output_steps, dt)
  outputs = jnp.einsum('ij,j...->i...',
                       jnp.linalg.pinv(output_phases.T),
                       fields)
  return outputs[:omega.shape[0]] + 1j * outputs[omega.shape[0]:]


def _prop_axis(mode):
  # TODO: Move to a utils file, or better yet, move to modes with the full 3
  # components.
  if mode.shape[1:].count(1) == 1:
    return "xyz"[mode.shape[1:].index(1)]
  else:
    raise ValueError(
        f"``mode.shape[1:] must contain exactly one value of ``1``, "
        f"instead got ``mode.shape == {{mode.shape}}``.")


def _transverse_slice(arr, pos, axis):
  def is_axis(i):
    return 3 - (arr.ndim - i) == "xyz".find(axis)

  component_inds = [i for i in range(3) if i != "xyz".find(axis)]
  arr = arr[..., component_inds, :, :, :]
  return jax.lax.dynamic_slice(
      arr,
      start_indices=[pos if is_axis(i) else 0 for i in range(arr.ndim)],
      slice_sizes=[1 if is_axis(i) else arr.shape[i] for i in range(arr.ndim)],
  )


def _source(mode, pos, epsilon):
  return mode / _transverse_slice(epsilon, pos, _prop_axis(mode))


def _overlap(mode, pos, output):
  return jnp.sum(mode * _transverse_slice(output, pos, _prop_axis(mode)),
                 axis=(-4, -3, -2, -1))


def _scatter_impl(epsilon, omega, modes, pos, sim_params):
  sim = partial(field, epsilon=epsilon, omega=omega, sim_params=sim_params)

  # Simulation output fields.
  fields = [sim(source=_source(m, p, epsilon), source_pos=p)
            for (m, p) in zip(modes, pos)]

  # Scattering values.
  svals = [[_overlap(m, p, f) for f in fields] for (m, p) in zip(modes, pos)]

  # Gradient of scattering values w.r.t. ``epsilon``.
  grads = [[fi * fj for fj in fields] for fi in fields]

  return svals, grads, fields


def _scatter_fwd(epsilon, omega, modes, pos, sim_params):
  svals, grads, _ = _scatter_impl(
      epsilon, omega, modes, pos, sim_params)
  return svals, grads


def _scatter_bwd(pos, sim_params, grad, g):
  gradient = sum(
      sum(jnp.sum(jnp.real(gij[:, None, None, None, None] * gradij), axis=0)
          for gradij, gij in zip(gradi, gi))
      for gradi, gi in zip(grad, g))
  return gradient, None, None


# TODO: Either move ``pos`` into ``modes``, but let's not have to static it
# anymore.
@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
@partial(jax.jit, static_argnums=(3, 4))
def scatter(
    epsilon: jax.Array,
    omega: jax.Array,
    modes: Tuple[jax.Array],
    pos: Tuple[int],
    sim_params: SimParams,
):
  """Differentiable time-harmonic scattering values between ``modes``.

  Args:
    epsilon: ``(3, xx, yy, zz)`` array of permittivity values. Differentiable.
    omega: ``(ww,)`` array of angular frequencies.
    modes: ``(2, xx, yy, zz)`` arrays with exactly one singular spatial
      dimension identifying the "ports" to be used when computing scattering
      parameters.
    pos: Integers denoting the location of modes along their respective
      propagation axes.
    sim_params: Simulation parameters.

  Returns:
    Scattering values as ``svals[i][j]`` nested lists of ``(ww,)`` arrays
    containing the scattering values from mode ``i`` to mode ``j`` over angular
    frequencies ``omega``.

  """
  svals, _, _ = _scatter_impl(epsilon, omega, modes, pos, sim_params)
  return svals


scatter.defvjp(_scatter_fwd, _scatter_bwd)
