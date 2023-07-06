"""Define inputs and outputs."""

import jax

from typing import Any, NamedTuple, Optional, Tuple


class Port(NamedTuple):
  """Input/output ports.

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
  field: jax.Array
  wavevector: jax.Array
  position: Tuple[Any, Any, Any]
  is_input: Any = False
  is_output: Any = False


def waveguide_port(
        epsilon: jax.Array,
        omega: jax.Array,
        position: Tuple[Any, Any, Any],
        width: Tuple[int, int, int],
        mode_order: int,
        lobpcg_init_fields: Optional[jax.Array] = None,
        lobpcg_max_iters: int = 100,
        lobpcg_tol: Optional[float] = None,
        reference_fields: Optional[jax.Array] = None,
        is_input: bool = False,
        is_output: bool = False,
) -> Port:
  """Waveguide modes.

  Args:
    epsilon: Array of current permittivity values with
      ``epsilon.shape[-4:] == (3, xx, yy, zz)``.
    omega: Array of angular frequencies at which to solve for waveguide modes.
    position: ``(x0, y0, z0)`` position for computing modes within ``epsilon``.
    width: ``(xx, yy, zz)`` number of cells in which to form a window to compute
      mode fields. Mode computation will occur within cell indices within
      `(x0, y0, z0)` to `(x0 + xx - 1, y0 + yy - 1, z0 + zz - 1)` inclusive.
    mode_order: Positive integer denoting mode number to solve for, where the
      fundamental mode corresponds to ``mode_order == 1``.
    lobpcg_init_fields: ``Port.field`` array to use as initial search
      directions in the underlying LOBPCG call.
    lobpcg_max_iters: Maximum number of iterations for underlying LOBPCG call.
    lobpcg_tol: Error threshold for underlying LOBPCG call.
    is_input: See ``Port.is_input``.
    is_output: See ``Port.is_output``.

  Returns:
    ``Port`` object with ``field`` and ``wavevector`` having batch dimensions
    derived from the broadcasting of ``epsilon`` and ``omega`` batch dimensions.
  """
  pass
