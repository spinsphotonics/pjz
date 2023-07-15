"""Solve Maxwell's equations."""

import jax

from typing import List, Tuple


def scatter(
    epsilon: jax.Array,
    ports: List[Tuple[jax.Array, jax.Array, Tuple[int, int, int]]],
    input_waveform: jax.Array,
    output_coeffs: jax.Array,
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
  """Computes scattering parameters in a differentiable manner.

  ``scatter()`` wraps the ``fdtdz_jax.fdtdz()`` and serves as the primary
  interface for ``pjz`` in the sense that most of the other tools are designed 
  to either compute/manipulate the inputs to ``scatter()`` or else to process
  its outputs.

  ``scatter()`` allows the user to specify the spatial, input/output, and
  temporal dimensions via the ``epsilon``, ``ports``, and
  ``input_waveform``/``output_transform`` parameters respectively;
  with the returned scattering parameters intended to be used within loss or
  objective functions for optimization purposes.

  Critically, these parameters also admit leading batch dimensions (which are
  carried on to the output scattering values) that are intended to allow for a
  fuller description of the optimization problem while also allowing for
  parallelization across multiple hosts/GPUs.

  Args:
    epsilon: Array of permittivity values where
      ``epsilon.shape[-4:] == (3, xx, yy, zz)``.
    ports: Sequence of ``pp`` tuples of the form
      ``(excitation, wavevector, position)`` denoting the field profile,
      wavevector, and ``(x0, y0, z0)`` position of each port, where
      ``excitation.shape[-4:] == (2, xx0, yy0, zz0)`` with one of either
      ``xx0``, ``yy0``, or ``zz0`` being equal to ``1``.
    input_waveform: Source excitation coefficients with
      ``input_waveform.shape[-1] == tt``.
    output_transform: Complex-valued array with
      ``output_transform.shape[-2:] == (vv, ww)`` used to convert ``vv``
      temporal output snapshots to ``ww`` field patterns.
    dt: (Dimensionless) time step, see ``fdtdz_jax.fdtdz()`` documentation.
    output_steps: ``(start, stop, step)`` tuple of integers denoting update
      steps at which to output fields.
    launch_params: See ``fdtdz_jax.fdtdz()`` documentation.
    absorption_padding: Padding cells to add along both boundaries of the x- and
      y-axes for adiabatic absorption boundary conditions.
    absorption_strength: Scaling coefficient for adiabatic absorption boundary.
    pml_widths: See ``fdtdz_jax.fdtdz()`` documentation.
    pml_alpha_coeff: Constant value for ``pml_alpha`` parameter of
      ``fdtdz_jax.fdtdz()``.
    pml_sigma_lnr: Natural logarithm of PML reflectivity.
    pml_sigma_m: Exponent for spatial scaling of PML.
    use_reduced_precision: See ``fdtdz_jax.fdtdz()`` documentation.

  Returns:
    Nested list of complex-valued scattering matrix coefficients where
    ``s[i][j]`` corresponds to scattering from port ``i`` to port ``j``
    and has shape equal to the batch dimensions of the various input parameters
    broadcast together.

  """
  pass
