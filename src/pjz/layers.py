"""Compute Yee-cell permittivity values based on layered profiles.

`layer.render()` computes individual permittivity values for a
finite-difference grid based on the Yee cell in a differentiable way.

"""

import jax.numpy as jnp


def _render_single(layers, layer_pos, grid_start, grid_end, m, axis):
  # In-plane offsets.
  if axis != "x":
    layers = jnp.pad(layers[:, :-m, :], ((0, 0), (m, 0), (0, 0)), "edge")
  if axis != "y":
    layers = jnp.pad(layers[:, :, :-m], ((0, 0), (0, 0), (m, 0)), "edge")

  # Offsets along z-axis.
  if axis == "z":
    grid_start = grid_start[:, 1]
    grid_end = grid_end[:, 1]
  else:
    grid_start = grid_start[:, 0]
    grid_end = grid_end[:, 0]

  # Convert to "layer-chunked" form, which consists of 2m x 2m tiles.
  lc = jnp.reshape(layers, (layers.shape[0],
                            layers.shape[1] // (2 * m), 2 * m,
                            layers.shape[2] // (2 * m), 2 * m))

  # Weighting values for pixels within each tile.
  w = (jnp.arange(2 * m) - (m - 0.5)) / (2 * m)**2

  # Use `2 * m` factor instead of summing along one axis and averaging on the
  # other. Factor of `12` corresponds to the integral of `u**2` in the
  # denominator for grid size of `1`.
  grads = [jnp.mean(12 * (2 * m) * x, (2, 4))
           for x in (lc * w[:, None, None], lc * w)]

  # Per-layer average.
  avg = jnp.mean(lc, (2, 4))

  # Per-layer average of inverse.
  aoi = jnp.mean(1 / lc, (2, 4))

  # Get start and end points for each layer in each grid cell.
  # `p0` and `p1` have shape `(ll, zz)`.
  #
  # And -infinity and +infinity as start and end points for first and last
  # layers respectively.
  #
  # Clip the start and end point for each layer in each cell at the cell
  # boundaries.
  #
  p0, p1 = [jnp.clip(x[:, None], grid_start, grid_end) for x in
            (jnp.concatenate([jnp.array([-jnp.inf]), layer_pos]),
             jnp.concatenate([layer_pos, jnp.array([jnp.inf])]))]

  # Ratio of cell occupied by each layer.
  u = (p1 - p0) / (grid_end - grid_start)

  # Average position of each layer inside each cell with respect to cell center.
  z = (p0 + p1) / 2 - (grid_start + grid_end) / 2

  # Reduces across the layer dimension.
  def cross(x, y): return jnp.einsum("lxy,lz->xyz", x, y)

  # Average of inverse across all layers.
  aoi = cross(aoi, u)

  # Inverse of average across all layers.
  ioa = 1 / cross(avg, u)

  # Average of gradient across all layers.
  grads = [cross(g, u) for g in grads]

  # Get the z-gradient.
  #
  # Denominator has only a the cell size to the power of `2` because there is
  # already a factor in `u`.
  #
  grads.append(cross(avg, u * z) / ((grid_end - grid_start)**2 / 12))

  # Diagonal term of the projection matrix.
  sum_of_gradients = sum(g**2 for g in grads)
  pii = (grads["xyz".index(axis)]**2 /
         jnp.where(sum_of_gradients == 0, 1, sum_of_gradients))

  return 1 / (pii * aoi + (1 - pii) * ioa)


def render(layers, layer_pos, grid_start, grid_end, m):
  """Render a three-dimensional vector array of permittivity values.

  Produces a 3D vector array of permittivity values on the Yee cell based on a
  layered stack of 2D profiles at magnification `2 * m`. Along the z-axis, both
  the layer boundaries and grid positions are allowed to vary continuously,
  while along the x- and y-axes the size of each (unmagnified) cell is assumed
  to be `1`.

  Attempts to follow [1] but only computes the on-diagonal elements of the
  projection matrix and is adapted to a situation where there are no explicit
  interfaces because the pixel values are allowed to vary continuously within
  each layer.

  Instead, the diagonal elements of the projection matrix for a given subvolume
  are estimated by computing gradients across it where `df(u)/du` is computed as
  the integral of `f(u) * u` over the integral of `u**2`  where `u` is relative
  to the center of the cell.

  [1] Farjadpour, Ardavan, et al. "Improving accuracy by subpixel smoothing in
      the finite-difference time domain." Optics letters 31.20 (2006): 2972-2974

  Args:
    layers: `(ll, 2 * m * xx, 2 * m * yy)` array of magnified layer profiles.
    layer_pos: `(ll - 1)` array of interface positions between the `ll` layer.
      Assumed to be in monotonically increasing order.
    grid_start: `(zz, 2)` array of start positions along the z-axis for both
      Ex/Ey cells and offset Ez cells, in that order.
    grid_end: Same as `grid_start` but for the end positions of cells along z.
    m: Denotes a `2 * m` in-plane magnification factor of layer profiles.

  Returns:
    `(3, xx, yy, zz)` array of permittivity values with offsets and vector
    components according to the finite-difference Yee cell.

  """
  return jnp.stack(
      [_render_single(layers, layer_pos, grid_start, grid_end, m, axis)
       for axis in "xyz"])
