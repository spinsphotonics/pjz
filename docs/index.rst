**pjz** 😴👕👖💤: Photonics on JAX
==================================

`pjz <https://www.github.com/spinsphotonics/pjz>`_ (pee-jays) is 
`JAX <https://www.github.com/google/jax>`_ and 
`fdtd-z <https://www.github.com/spinsphotonics/fdtdz>`_, 
a set of tools for runnning photonic simulation and optimization workflows at
scale.

.. currentmodule:: pjz

.. automodule:: pjz

Solve API
---------
.. autofunction:: scatter

Ports API
---------
.. autoclass:: Port
.. autofunction :: waveguide_port

Frequency API
-------------
.. autofunction :: ramped_sin
.. autofunction :: sampling_interval
.. autofunction :: output_transform

Structure API
-------------
.. autofunction :: rect
.. autofunction :: circ 
.. autofunction :: invert 
.. autofunction :: union
.. autofunction :: intersect
.. autofunction :: dilate
.. autofunction :: shift
.. autofunction :: render_layers

Developer notes 
---------------


