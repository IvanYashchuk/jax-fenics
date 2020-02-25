import fdm
import jax
from jax.config import config
import jax.numpy as np

import fenics
import ufl

from jaxfenics import build_fem_eval

config.update("jax_enable_x64", True)
fenics.parameters["std_out_all_processes"] = False
fenics.set_log_level(fenics.LogLevel.ERROR)

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
jax_fem_eval = build_fem_eval(solve_fenics, templates)

# multivariate output function
ff = lambda x, y: np.sqrt(np.square(jax_fem_eval(np.sqrt(x ** 3), y)))  # noqa: E731
x_input = np.ones(1)
y_input = 1.2 * np.ones(1)

# multivariate output function of the first argument
hh = lambda x: ff(x, y_input)  # noqa: E731
# multivariate output function of the second argument
gg = lambda y: ff(x_input, y)  # noqa: E731

fdm_jac0 = fdm.jacobian(hh)(x_input)
jax_jac0 = jax.jacrev(hh)(x_input)


def test_jacobian0():
    assert np.allclose(fdm_jac0, jax_jac0)


# random vec for vjp test
rngkey = jax.random.PRNGKey(0)
v = jax.random.normal(rngkey, shape=(V.dim(),), dtype="float64")

fdm_vjp0 = v @ fdm_jac0
jax_vjp0 = jax.vjp(hh, x_input)[1](v)


def test_vjp0():
    assert np.allclose(fdm_vjp0, jax_vjp0)


fdm_jac1 = fdm.jacobian(gg)(y_input)
jax_jac1 = jax.jacrev(gg)(y_input)


def test_jacobian1():
    assert np.allclose(fdm_jac1, jax_jac1)


# random vec for vjp test
rngkey = jax.random.PRNGKey(1)
v = jax.random.normal(rngkey, shape=(V.dim(),), dtype="float64")

fdm_vjp1 = v @ fdm_jac1
jax_vjp1 = jax.vjp(gg, y_input)[1](v)


def test_vjp1():
    assert np.allclose(fdm_vjp1, jax_vjp1)


# scalar output function
f_scalar = lambda x, y: np.sqrt(  # noqa: E731
    np.sum(np.square(jax_fem_eval(np.sqrt(x ** 3), y)))
)
h_scalar = lambda x: f_scalar(x, y_input)  # noqa: E731

fdm_grad = fdm.gradient(h_scalar)(x_input)
jax_grad = jax.grad(h_scalar)(x_input)


def test_grad():
    assert np.allclose(fdm_grad, jax_grad)


jax_grads = jax.grad(f_scalar, (0, 1))(x_input, y_input)
fdm_grad0 = fdm_grad
fdm_grad1 = fdm.gradient(lambda y: f_scalar(x_input, y))(y_input)  # noqa: E731


def test_grad_multiple_input0():
    assert np.allclose(fdm_grad0, jax_grads[0])


def test_grad_multiple_input1():
    assert np.allclose(fdm_grad1, jax_grads[1])
