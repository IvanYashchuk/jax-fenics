import fdm
import jax
from jax.config import config
import jax.numpy as np

import fenics
import ufl

from jaxfenics import fem_eval, vjp_dfem_impl, build_fem_eval

config.update("jax_enable_x64", True)

mesh = fenics.UnitSquareMesh(6, 5)
V = fenics.FunctionSpace(mesh, "P", 1)


def solve_fenics(kappa0, kappa1):

    f = fenics.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    u = fenics.Function(V)
    bc = fenics.DirichletBC(V, fenics.Constant(0.0), "on_boundary")

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    JJ = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    v = fenics.TestFunction(V)
    F = fenics.derivative(JJ, u, v)
    fenics.solve(F == 0, u, bcs=bc)
    return u, F


templates = (fenics.Constant(0.0), fenics.Constant(0.0))
inputs = (np.ones(1) * 0.5, np.ones(1) * 0.6)


def test_fenics_vjp():
    numpy_output, fenics_solution, residual_form, fenics_inputs = fem_eval(
        solve_fenics, templates, *inputs
    )
    g = np.ones_like(numpy_output)
    jax_grad_tuple = vjp_dfem_impl(g, fenics_solution, residual_form, fenics_inputs)
    assert np.isclose(jax_grad_tuple[0], np.asarray(-2.91792642)) and np.isclose(
        jax_grad_tuple[1], np.asarray(2.43160535)
    )
