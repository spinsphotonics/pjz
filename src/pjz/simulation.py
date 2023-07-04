"""Simulation."""

from typing import NamedTuple


def scatter(
    epsilon,
    ports,
    input_waveform,
    output_coeffs,
    compress_batch_dims,
    dt,
    output_steps,
    launch_params,
    absorption_padding=40,
    absorption_coeff=1e-2,
    pml_widths=(8, 8),
    pml_alpha_coeff=0.05,
    pml_sigma_lnr=16.0,
    pml_sigma_m=4.0,
    use_reduced_precision=True,
):
  """Computes scattering parameters, differentiable.

  Note that this uses the average of the ``ww`` dimension of ``ports[i].field``
  as individual excitations...
  
  Sounds like some pmap or vmap should be able to be used here??

  Args:
    epsilon: Array of permittivity values where
      ``epsilon.shape[-4:] == (3, xx, yy, zz)``.
    ports: List of ``Port`` objects.
    input_waveform: Source excitation coefficients with
      ``input_waveform.shape[-1] == tt``.
    output_transform: Complex-valued array with
      ``output_transform.shape[-2:] == (vv, ww)`` used to convert ``vv``
      temporal output snapshots to ``ww`` field patterns.
    compress_batch_dims: Tuple of batch dimensions across ``epsilon``,
      ``ports[i].field``, and ``input_waveform`` to run as a single simulation.
    dt: (Dimensionless) time step, see ``fdtdz_jax.fdtdz()`` documentation.
    output_steps: ``(start, stop, step)`` tuple of integers denoting update
      steps at which to output fields.
    launch_params: See ``fdtdz_jax.fdtdz()`` documentation.
    absorption_padding: Padding cells to add along both coundaries of the x- and
      y-axes for adiabatic absorption boundary conditions.
    absorption_strength: Scaling coefficient for adiabatic absorption boundary.
    pml_widths: See ``fdtdz_jax.fdtdz()`` documentation.
    pml_alpha_coeff: Constant value for ``pml_alpha`` parameter of ``fdtdz_jax.fdtdz()``.
    pml_sigma_lnr: Natural logarithm of PML reflectivity.
    pml_sigma_m: Exponent for spatial scaling of PML.
    use_reduced_precision: See ``fdtdz_jax.fdtdz()`` documentation.

  Returns:
    Array of complex-valued scattering matrix coefficients with
    ``shape[:-2]`` equal to ``epsilon``, ``ports[i].field``, ``input_waveform``,
    and ``output_transform`` batch dimensions broadcast together and 
    ``shape[-2:] == (mm, nn)`` for ``mm`` input and ``nn`` output ports.

  """
  pass


class Port(NamedTuple):
  """Input/output port for ``scatter()``.

  Note that this also supports a "point" source, but the non-point sources must be on a border...


  Attributes:
    field: Array of field values where ``field.shape[-4:] == (3, xx, yy, zz)``.
      For compatibility with ``fdtdz_jax.fdtdz()``, which only implements source
      fields in a plane, the elements at ``(... , i, :, :, :)`` must be non-zero
      for ``i`` corresponding to one of either ``xx``, ``yy``, or ``zz`` being
      set to ``1``.
    wavevector: Array of wavevector values corresponding with
      ``wavevector.shape == field.shape[:-4]``.
    position: ``(x0, y0, z0)`` tuple denoting port position. Values of
      ``+jax.np.inf`` and ``-jax.np.inf`` are used to denote a position on a
      boundary. Fields are understood to be extend from indices ``(x0, y0, z0)``
      to ``(x0 + xx - 1, y0 + yy - 1, z0 + zz - 1)`` inclusive.
    is_input: If `True`, denotes an input port.
    is_output: If `True`, denotes an output port.

  """
  pass


def scatterport(
    omega,
    epsilon,
    omega_ref,
    epsilon_ref,
    center,
    width,
    is_input=False,
    is_output=False,
    subspace_iters=5,
):
  """Same as 
  """
