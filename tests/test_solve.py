import pytest

import jax
from jax.config import config
import jax.numpy as np
import numpy as onp

import fenics
import ufl

import fdm

from jaxfenics import solve_eval, vjp_solve_eval_impl
from jaxfenics import jvp_solve_eval
from jaxfenics import fenics_to_numpy, numpy_to_fenics

config.update("jax_enable_x64", True)

mesh = fenics.UnitSquareMesh(6, 5)
V = fenics.FunctionSpace(mesh, "P", 1)


def solve_fenics(kappa0, kappa1):

    f = fenics.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    u = fenics.Function(V)
    bcs = [fenics.DirichletBC(V, fenics.Constant(0.0), "on_boundary")]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    JJ = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    v = fenics.TestFunction(V)
    F = fenics.derivative(JJ, u, v)
    fenics.solve(F == 0, u, bcs=bcs)
    return u, F, bcs


templates = (fenics.Constant(0.0), fenics.Constant(0.0))
inputs = (np.ones(1) * 0.5, np.ones(1) * 0.6)


def test_fenics_forward():
    numpy_output, _, _, _, _ = solve_eval(solve_fenics, templates, *inputs)
    u, _, _ = solve_fenics(fenics.Constant(0.5), fenics.Constant(0.6))
    assert np.allclose(numpy_output, fenics_to_numpy(u))


def test_fenics_forward_wrong_form():
    def solve_fenics(kappa0, kappa1):
        u = fenics.Function(V)
        v = fenics.TestFunction(V)
        inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
        JJ = 0.5 * inner(kappa0 * grad(u), grad(v)) * dx  # J is linear form
        F = fenics.derivative(JJ, u, fenics.TrialFunction(V))  # F is now bilinear form
        bcs = [fenics.DirichletBC(V, fenics.Constant(0.0), "on_boundary")]
        return u, F, bcs

    with pytest.raises(ValueError):
        solve_eval(solve_fenics, templates, *inputs)


def test_fenics_vjp():
    numpy_output, fenics_solution, residual_form, fenics_inputs, bcs = solve_eval(
        solve_fenics, templates, *inputs
    )
    g = np.ones_like(numpy_output)
    jax_grad_tuple = vjp_solve_eval_impl(
        g, fenics_solution, residual_form, fenics_inputs, bcs
    )
    check1 = np.isclose(jax_grad_tuple[0], np.asarray(-2.91792642))
    check2 = np.isclose(jax_grad_tuple[1], np.asarray(2.43160535))
    assert check1 and check2


def test_fenics_jvp():
    primals = inputs
    tangent0 = np.asarray(onp.random.normal(size=(1,)))
    tangent1 = np.asarray(onp.random.normal(size=(1,)))
    tangents = (tangent0, tangent1)

    ff0 = lambda x: solve_eval(solve_fenics, templates, x, primals[1])[0]  # noqa: E731
    ff1 = lambda y: solve_eval(solve_fenics, templates, primals[0], y)[0]  # noqa: E731
    fdm_jvp0 = fdm.jvp(ff0, tangents[0])(primals[0])
    fdm_jvp1 = fdm.jvp(ff1, tangents[1])(primals[1])

    _, out_tangent = jvp_solve_eval(solve_fenics, templates, primals, tangents)

    assert np.allclose(fdm_jvp0 + fdm_jvp1, out_tangent)
