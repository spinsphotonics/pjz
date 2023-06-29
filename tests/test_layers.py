import pjz

import jax
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize("domain,m,vals", [((4, 5, 6), 1, (1, 3))])
@pytest.mark.parametrize("thresh,index,expected", [
    pytest.param((5, 0, -0.5), (0, 1, 0, 0), 1.0, id="x-Ex-pre"),
    pytest.param((5, 0, -0.5), (0, 2, 0, 0), 1.5, id="x-Ex-on"),
    pytest.param((5, 0, -0.5), (0, 3, 0, 0), 3.0, id="x-Ex-post"),
    pytest.param((4, 0, -0.5), (1, 1, 0, 0), 1.0, id="x-Ey-pre"),
    pytest.param((4, 0, -0.5), (1, 2, 0, 0), 2.0, id="x-Ey-on"),
    pytest.param((4, 0, -0.5), (1, 3, 0, 0), 3.0, id="x-Ey-post"),
    pytest.param((0, 5, -0.5), (1, 0, 1, 0), 1.0, id="y-Ey-pre"),
    pytest.param((0, 5, -0.5), (1, 0, 2, 0), 1.5, id="y-Ey-on"),
    pytest.param((0, 5, -0.5), (1, 0, 3, 0), 3.0, id="y-Ey-post"),
    pytest.param((0, 4, -0.5), (2, 0, 1, 0), 1.0, id="y-Ez-pre"),
    pytest.param((0, 4, -0.5), (2, 0, 2, 0), 2.0, id="y-Ez-on"),
    pytest.param((0, 4, -0.5), (2, 0, 3, 0), 3.0, id="y-Ez-post"),
    pytest.param((0, 0, +2.5), (2, 0, 0, 1), 1.0, id="z-Ez-pre"),
    pytest.param((0, 0, +2.5), (2, 0, 0, 2), 1.5, id="z-Ez-on"),
    pytest.param((0, 0, +2.5), (2, 0, 0, 3), 3.0, id="z-Ez-post"),
    pytest.param((0, 0, +2.0), (1, 0, 0, 1), 1.0, id="z-Ex-pre"),
    pytest.param((0, 0, +2.0), (1, 0, 0, 2), 2.0, id="z-Ex-on"),
    pytest.param((0, 0, +2.0), (1, 0, 0, 3), 3.0, id="z-Ex-post"),
])
def test_simple(domain, m, vals, thresh, index, expected):
  xx, yy, zz = domain
  layer = vals[0] * jnp.ones((2, 2 * m * xx, 2 * m * yy))
  layer = layer.at[1, thresh[0]:, thresh[1]:].set(vals[1])
  out = pjz.render(layer,
                   jnp.array([thresh[2]]),
                   jnp.arange(zz)[:, None] + jnp.array([[-0.5, 0]]),
                   jnp.arange(zz)[:, None] + jnp.array([[0.5, 1]]),
                   m)
  assert out[index] == pytest.approx(expected)


@pytest.mark.parametrize("m,thresh,component,index,expected", [
    # Check threshold values as `m` increases for Ex.
    (1, 3, 0, 0, 1.0),
    (1, 3, 0, 1, 1.5),
    (1, 3, 0, 2, 3.0),
    (2, 6, 0, 0, 1.0),
    (2, 6, 0, 1, 1.5),
    (2, 6, 0, 2, 3.0),
    (3, 9, 0, 0, 1.0),
    (3, 9, 0, 1, 1.5),
    (3, 9, 0, 2, 3.0),
    # Check interpolated values as threshold sweeps through cell.
    (3, 6, 0, 1, 3.0),
    (3, 7, 0, 1, 9 / 4),
    (3, 8, 0, 1, 9 / 5),
    (3, 9, 0, 1, 1.5),
    (3, 10, 0, 1, 9 / 7),
    (3, 11, 0, 1, 9 / 8),
    (3, 12, 0, 1, 1.0),
    # Check threshold values as `m` increases for Ey.
    (1, 4, 1, 1, 1.0),
    (1, 4, 1, 2, 2.0),
    (1, 4, 1, 3, 3.0),
    (2, 8, 1, 1, 1.0),
    (2, 8, 1, 2, 2.0),
    (2, 8, 1, 3, 3.0),
    (3, 12, 1, 1, 1.0),
    (3, 12, 1, 2, 2.0),
    (3, 12, 1, 3, 3.0),
    # Check interpolated values as threshold sweeps through cell.
    (3, 9, 1, 2, 3.0),
    (3, 10, 1, 2, 8 / 3),
    (3, 11, 1, 2, 7 / 3),
    (3, 12, 1, 2, 2.0),
    (3, 13, 1, 2, 5 / 3),
    (3, 14, 1, 2, 4 / 3),
    (3, 15, 1, 2, 1.0),
])
def test_x_threshold(m, thresh, component, index, expected):
  xx, yy, zz = 4, 3, 2
  vals = (1, 3)
  layer = vals[0] * jnp.ones((1, 2 * m * xx, 2 * m * yy))
  layer = layer.at[0, thresh:, :].set(vals[1])
  out = pjz.render(layer,
                   jnp.array([]),
                   jnp.arange(zz)[:, None] + jnp.array([[-0.5, 0]]),
                   jnp.arange(zz)[:, None] + jnp.array([[0.5, 1]]),
                   m)
  assert out[component, index, 0, 0] == pytest.approx(expected)


@pytest.mark.parametrize("thresh,component,index,expected", [
    # Check values for `Ez` as we sweep the threshold.
    (0.00, 2, 0, 3.0),
    (0.25, 2, 0, 2.0),
    (0.50, 2, 0, 1.5),
    (0.75, 2, 0, 1.2),
    (1.00, 2, 0, 1.0),
    # Check values for `Ex` as we sweep the threshold.
    (-0.50, 0, 0, 3.0),
    (-0.25, 0, 0, 2.5),
    (+0.00, 0, 0, 2.0),
    (+0.25, 0, 0, 1.5),
    (+0.50, 0, 0, 1.0),
])
def test_z_threshold(thresh, component, index, expected):
  xx, yy, zz = 1, 1, 1
  vals = (1, 3)
  layer = vals[0] * jnp.ones((2, 2 * xx, 2 * yy))
  layer = layer.at[1].set(vals[1])
  out = pjz.render(layer,
                   jnp.array([thresh]),
                   jnp.arange(zz)[:, None] + jnp.array([[-0.5, 0]]),
                   jnp.arange(zz)[:, None] + jnp.array([[0.5, 1]]),
                   m=1)
  assert out[component, 0, 0, index] == pytest.approx(expected)


# Tests "corner" cases for all components. Importantly shows equivalency
# (at this level of magnification at least) between thresholds in all
# three axes directions.
@pytest.mark.parametrize("thresh,index,expected", [
    pytest.param((0, 0, -0.5), (0, 0, 1, 0), 3.0, id="Ex-post-all"),
    pytest.param((1, 0, -0.5), (0, 0, 1, 0), 1.5, id="Ex-on-x"),
    pytest.param((2, 0, -0.5), (0, 0, 1, 0), 1.0, id="Ex-pre-x"),
    pytest.param((0, 2, -0.5), (0, 0, 1, 0), 2.0, id="Ex-on-y"),
    pytest.param((0, 3, -0.5), (0, 0, 1, 0), 1.0, id="Ex-pre-y"),
    pytest.param((0, 0, +0.0), (0, 0, 1, 0), 2.0, id="Ex-on-z"),
    pytest.param((0, 0, +0.5), (0, 0, 1, 0), 1.0, id="Ex-pre-z"),
    pytest.param((1, 2, -0.5), (0, 0, 1, 0),
                 1 / (0.5 * (5 / 6) + 0.5 * (2 / 3)), id="Ex-on-xy"),
    pytest.param((0, 2, +0.0), (0, 0, 1, 0), 1.5, id="Ex-on-yz"),
    pytest.param((1, 0, +0.0), (0, 0, 1, 0),
                 1 / (0.5 * (5 / 6) + 0.5 * (2 / 3)), id="Ex-on-xz"),
    pytest.param((1, 2, +0.0), (0, 0, 1, 0),
                 1 / ((1 / 3) * (11 / 12) + (2 / 3) * (4 / 5)), id="Ex-on-xyz"),
    pytest.param((0, 0, -0.5), (1, 1, 0, 0), 3.0, id="Ey-post-all"),
    pytest.param((2, 0, -0.5), (1, 1, 0, 0), 2.0, id="Ey-on-x"),
    pytest.param((3, 0, -0.5), (1, 1, 0, 0), 1.0, id="Ey-pre-x"),
    pytest.param((0, 1, -0.5), (1, 1, 0, 0), 1.5, id="Ey-on-y"),
    pytest.param((0, 2, -0.5), (1, 1, 0, 0), 1.0, id="Ey-pre-y"),
    pytest.param((0, 0, +0.0), (1, 1, 0, 0), 2.0, id="Ey-on-z"),
    pytest.param((0, 0, +0.5), (1, 1, 0, 0), 1.0, id="Ey-pre-z"),
    pytest.param((2, 1, -0.5), (1, 1, 0, 0),
                 1 / (0.5 * (5 / 6) + 0.5 * (2 / 3)), id="Ey-on-xy"),
    pytest.param((0, 1, +0.0), (1, 1, 0, 0),
                 1 / (0.5 * (5 / 6) + 0.5 * (2 / 3)), id="Ey-on-yz"),
    pytest.param((2, 0, +0.0), (1, 1, 0, 0), 1.5, id="Ey-on-xz"),
    pytest.param((2, 1, +0.0), (1, 1, 0, 0),
                 1 / ((1 / 3) * (11 / 12) + (2 / 3) * (4 / 5)), id="Ey-on-xyz"),
    pytest.param((0, 0, 0.0), (2, 1, 1, 0), 3.0, id="Ez-post-all"),
    pytest.param((2, 0, 0.0), (2, 1, 1, 0), 2.0, id="Ez-on-x"),
    pytest.param((3, 0, 0.0), (2, 1, 1, 0), 1.0, id="Ez-pre-x"),
    pytest.param((0, 2, 0.0), (2, 1, 1, 0), 2.0, id="Ez-on-y"),
    pytest.param((0, 3, 0.0), (2, 1, 1, 0), 1.0, id="Ez-pre-y"),
    pytest.param((0, 0, 0.5), (2, 1, 1, 0), 1.5, id="Ez-on-z"),
    pytest.param((0, 0, 1.0), (2, 1, 1, 0), 1.0, id="Ez-pre-z"),
    pytest.param((2, 2, 0.0), (2, 1, 1, 0), 1.5, id="Ez-on-xy"),
    pytest.param((0, 2, 0.5), (2, 1, 1, 0),
                 1 / (0.5 * (5 / 6) + 0.5 * (2 / 3)), id="Ez-on-yz"),
    pytest.param((2, 0, 0.5), (2, 1, 1, 0),
                 1 / (0.5 * (5 / 6) + 0.5 * (2 / 3)), id="Ez-on-xz"),
    pytest.param((2, 2, 0.5), (2, 1, 1, 0),
                 1 / ((1 / 3) * (11 / 12) + (2 / 3) * (4 / 5)), id="Ez-on-xyz"),
])
def test_corner(thresh, index, expected):
  xx, yy, zz = 2, 2, 2
  vals = (1, 3)
  layer = vals[0] * jnp.ones((2, 2 * xx, 2 * yy))
  layer = layer.at[1, thresh[0]:, thresh[1]:].set(vals[1])
  out = pjz.render(layer,
                   jnp.array([thresh[2]]),
                   jnp.arange(zz)[:, None] + jnp.array([[-0.5, 0]]),
                   jnp.arange(zz)[:, None] + jnp.array([[0.5, 1]]),
                   m=1)
  assert out[index] == pytest.approx(expected)


def test_differentiable():
  xx, yy, zz = 2, 2, 2
  grad = jax.grad(lambda *args: jnp.sum(pjz.render(*args)), (0, 1))
  layer_grad, pos_grad = grad(jnp.ones((2, 2 * xx, 2 * yy)),
                              jnp.array([1.0]),
                              jnp.arange(zz)[:, None] + jnp.array([[-0.5, 0]]),
                              jnp.arange(zz)[:, None] + jnp.array([[0.5, 1]]),
                              1)
  assert layer_grad.shape == (2, 2 * xx, 2 * yy)
  assert pos_grad.shape == (1,)
