"""Simulation."""

from collections import namedtuple


# TODO: Figure out how we can smush/unsmush (maybe even have a toggle for it?)
# how we do the frequency stuff.
def scatter(
    epsilon,
    ports,
    dt,
    input_waveform,
    output_steps,
    output_coeffs,
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

  Args:
    epsilon: ``(3, xx, yy, zz)`` array of permittivity values.
    ports: List of ``pp`` ports as returned by ``scatterport()``.
    dt: (Dimensionless) time step, see ``fdtdz_jax.fdtdz()`` documentation.
    input_waveform: ``(tt,)`` array of source excitation coefficients.
    output_steps: ``(start, stop, step)`` tuple of integers denoting update
      steps at which to output fields.
    output_coeffs: ``(vv, ww)`` complex-valued array used to linearly transform
      ``vv`` temporal output snapshots to ``ww`` field patterns.
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
    ``(mm, nn, ww)`` array of complex-valued scattering matrix coefficients for
    ``mm`` input and ``nn`` output ports.

  """
  pass


class ScatterPort (namedtuple(
        "ScatterPort", ["field", "wavevector", "position", "is_input", "is_output"])):
  """Input/output port for ``scatter()``.

  Attributes:
    field: ``(ww, 3, xx, yy, zz)`` array 
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
