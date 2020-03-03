# jax-fenics &middot; [![Build](https://github.com/ivanyashchuk/jax-fenics/workflows/CI/badge.svg)](https://github.com/ivanyashchuk/jax-fenics/actions?query=workflow%3ACI+branch%3Amaster) [![Coverage Status](https://coveralls.io/repos/github/IvanYashchuk/jax-fenics/badge.svg?branch=master)](https://coveralls.io/github/IvanYashchuk/jax-fenics?branch=master)

This package enables use of [FEniCS](http://fenicsproject.org) for solving differentiable variational problems in [JAX](https://github.com/google/jax).

Automatic tangent linear and adjoint solvers are implemented for FEniCS programs involving `fenics.solve` and `fenics.assemble`.
These solvers make it possible to use JAX's forward and reverse Automatic Differentiation with FEniCS.

Current limitations:
* Composition of forward and reverse modes for higher-order derivatives is not implemented yet.
* Differentiating through time-dependent FEniCS programs is not supported, for this case check out [jax-fenics-adjoint](https://github.com/IvanYashchuk/jax-fenics-adjoint) or [jax-firedrake](https://github.com/IvanYashchuk/jax-firedrake).
  * It is possible to write differentiable time-stepping in JAX, check the [Burgers' equation example](https://github.com/IvanYashchuk/jax-fenics/blob/master/examples/burgers.py).

## Example
Here is the demonstration of solving the [Poisson's PDE](https://en.wikipedia.org/wiki/Poisson%27s_equation)
on 2D square domain and calculating the solution Jacobian matrix (_du/df_) using the reverse (adjoint) mode Automatic Differentiation.
```python
import jax
import jax.numpy as np
from jax.config import config
config.update("jax_enable_x64", True)

import fenics
import ufl

from jaxfenics import build_jax_solve_eval
from jaxfenics import numpy_to_fenics

# Create mesh for the unit square domain
n = 10
mesh = fenics.UnitSquareMesh(n, n)

# Define discrete function spaces and functions
V = fenics.FunctionSpace(mesh, "CG", 1)
W = fenics.FunctionSpace(mesh, "DG", 0)

templates = (fenics.Function(W),)
@build_jax_solve_eval(templates)
def fenics_solve(f):
    u = fenics.Function(V, name="PDE Solution")
    v = fenics.TestFunction(V)
    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    F = (inner(grad(u), grad(v)) - f * v) * dx
    bcs = [fenics.DirichletBC(V, 0.0, "on_boundary")]
    fenics.solve(F == 0, u, bcs)
    # output should be a tuple (solution, F, bcs)
    return u, F, bcs

# build_jax_solve_eval is a wrapper decorator that registers
# `fenics_solve` for JAX and makes fenics_solve to return only the PDE solution

# Let's create a vector of ones with size equal to the number of cells in the mesh
f = np.ones(W.dim())
u = fenics_solve(f) # u is JAX's array
u_fenics = numpy_to_fenics(u, fenics.Function(V)) # we need to explicitly provide template function for conversion

# now we can calculate vector-Jacobian product with `jax.vjp`
jvp_result = jax.vjp(fenics_solve, f)[1](np.ones_like(u))

# or the full (dense) Jacobian matrix du/df with `jax.jacrev`
dudf = jax.jacrev(fenics_solve)(f)
# our function fenics_solve maps R^200 (dimension of W) to R^121 (dimension of V)
# therefore the Jacobian matrix dimension is dim V x dim W
assert dudf.shape == (V.dim(), W.dim())
```
Check `examples/` or `tests/` folders for the additional examples.

## Installation
First install [FEniCS](http://fenicsproject.org).
Then install [JAX](https://github.com/google/jax) with:

    python -m pip install pip install --upgrade jax jaxlib  # CPU-only version

After that install the jax-fenics with:

    python -m pip install git+https://github.com/IvanYashchuk/jax-fenics.git@master

## Reporting bugs

If you found a bug, create an [issue].

[issue]: https://github.com/IvanYashchuk/jax-fenics/issues/new

## Contributing

Pull requests are welcome from everyone.

Fork, then clone the repository:

    git clone https://github.com/IvanYashchuk/jax-fenics.git

Make your change. Add tests for your change. Make the tests pass:

    pytest tests/

Check the formatting with `black` and `flake8`. Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/IvanYashchuk/jax-fenics/pulls