import jax
from jax.config import config
import jax.numpy as np
import numpy as onp

import fenics
import ufl

import fdm

from jaxfenics import assemble_eval, vjp_assemble_eval, jvp_assemble_eval
from jaxfenics import fenics_to_numpy, numpy_to_fenics

config.update("jax_enable_x64", True)

mesh = fenics.UnitSquareMesh(3, 2)
V = fenics.FunctionSpace(mesh, "P", 1)


def assemble_fenics(u, kappa0, kappa1):

    f = fenics.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    J_form = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    J = fenics.assemble(J_form)
    return J, J_form


templates = (fenics.Function(V), fenics.Constant(0.0), fenics.Constant(0.0))
inputs = (np.ones(V.dim()), np.ones(1) * 0.5, np.ones(1) * 0.6)


def test_fenics_forward():
    numpy_output, _, _, = assemble_eval(assemble_fenics, templates, *inputs)
    u1 = fenics.interpolate(fenics.Constant(1.0), V)
    J, _ = assemble_fenics(u1, fenics.Constant(0.5), fenics.Constant(0.6))
    assert np.allclose(numpy_output, J)


def test_vjp_assemble_eval():
    numpy_output, vjp_fun = vjp_assemble_eval(assemble_fenics, templates, *inputs)
    g = np.ones_like(numpy_output)
    vjp_out = vjp_fun(g)

    ff = lambda *args: assemble_eval(assemble_fenics, templates, *args)[0]  # noqa: E731
    ff0 = lambda x: ff(x, inputs[1], inputs[2])  # noqa: E731
    ff1 = lambda y: ff(inputs[0], y, inputs[2])  # noqa: E731
    ff2 = lambda z: ff(inputs[0], inputs[1], z)  # noqa: E731
    fdm_jac0 = fdm.jacobian(ff0)(inputs[0])
    fdm_jac1 = fdm.jacobian(ff1)(inputs[1])
    fdm_jac2 = fdm.jacobian(ff2)(inputs[2])

    check1 = np.allclose(vjp_out[0], fdm_jac0)
    check2 = np.allclose(vjp_out[1], fdm_jac1)
    check3 = np.allclose(vjp_out[2], fdm_jac2)
    assert check1 and check2 and check3


def test_jvp_assemble_eval():
    primals = inputs
    tangent0 = np.asarray(onp.random.normal(size=(V.dim(),)))
    tangent1 = np.asarray(onp.random.normal(size=(1,)))
    tangent2 = np.asarray(onp.random.normal(size=(1,)))
    tangents = (tangent0, tangent1, tangent2)

    ff = lambda *args: assemble_eval(assemble_fenics, templates, *args)[0]  # noqa: E731
    ff0 = lambda x: ff(x, primals[1], primals[2])  # noqa: E731
    ff1 = lambda y: ff(primals[0], y, primals[2])  # noqa: E731
    ff2 = lambda z: ff(primals[0], primals[1], z)  # noqa: E731
    fdm_jvp0 = fdm.jvp(ff0, tangents[0])(primals[0])
    fdm_jvp1 = fdm.jvp(ff1, tangents[1])(primals[1])
    fdm_jvp2 = fdm.jvp(ff2, tangents[2])(primals[2])

    _, out_tangent = jvp_assemble_eval(assemble_fenics, templates, primals, tangents)

    assert np.allclose(fdm_jvp0 + fdm_jvp1 + fdm_jvp2, out_tangent)
