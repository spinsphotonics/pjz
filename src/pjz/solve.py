"""Solve Maxwell's equations."""

from . import ports

import jax

from typing import List, Tuple


def scatter(
    epsilon: jax.Array,
    ports: List[ports.Port],
    input_waveform: jax.Array,
    output_coeffs: jax.Array,
    compress_batch_dims: Tuple[int, ...],
    dt: float,
    output_steps: Tuple[int, int, int],
    launch_params: Tuple[Tuple[int, int], Tuple[int, int], int, Tuple[int, int]],
    absorption_padding: int = 40,
    absorption_coeff: float = 1e-2,
    pml_widths: Tuple[int, int] = (8, 8),
    pml_alpha_coeff: float = 0.05,
    pml_sigma_lnr: float = 16.0,
    pml_sigma_m: float = 4.0,
    use_reduced_precision: bool = True,
) -> jax.Array:
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
