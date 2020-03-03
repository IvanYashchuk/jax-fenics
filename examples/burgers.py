# Solves time-dependent 1D Burgers' equation and evaluates gradient of energy wrt initial conditions.
# This example is based on
# https://github.com/dolfin-adjoint/pyadjoint/blob/master/examples/burgers/burgers.py

import jax
from jax.config import config

import jax.numpy as np
import numpy as onp

import fenics as fn
import ufl

from jaxfenics import build_jax_solve_eval, build_jax_assemble_eval
from jaxfenics import numpy_to_fenics, fenics_to_numpy

import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
fn.set_log_level(fn.LogLevel.ERROR)

# Create mesh
n = 30
mesh = fn.UnitIntervalMesh(n)

# Define discrete function spaces and functions
V = fn.FunctionSpace(mesh, "CG", 2)

v = fn.TestFunction(V)
nu = fn.Constant(0.0001)
a = fn.Constant(0.4)
timestep = np.asarray([0.05])
bcs = [fn.DirichletBC(V, 0.0, "on_boundary")]


def Dt(u, u_prev, timestep):
    return (u - u_prev) / timestep


solve_templates = (fn.Function(V), fn.Constant(0.0))
assemble_templates = (fn.Function(V),)


@build_jax_solve_eval(solve_templates)
def fenics_solve(u_prev, timestep):
    # Define and solve one step of the Burgers equation
    u = fn.Function(V)
    F = (
        Dt(u, u_prev, timestep) * v + a * u * u.dx(0) * v + nu * u.dx(0) * v.dx(0)
    ) * ufl.dx
    fn.solve(F == 0, u, bcs)
    return u, F, bcs


@build_jax_assemble_eval(assemble_templates)
def eval_functional(u):
    J_form = u * u * ufl.dx
    J = fn.assemble(J_form)
    return J, J_form


u0_fenics = fn.interpolate(fn.Expression("sin(2*pi*x[0])", element=V.ufl_element()), V)
u0 = fenics_to_numpy(u0_fenics)


def solve_burgers(timespan, u0):
    t, end = timespan
    J = 0
    u = u0
    while t <= end:
        u = fenics_solve(u, timestep)
        t += float(timestep)
        J += float(timestep) * eval_functional(u)
    return u, J


tspan = (0.0, 0.3)
u, J = solve_burgers(tspan, u0)

final_solution = numpy_to_fenics(u, fn.Function(V))

eval_J = lambda u0: solve_burgers(tspan, u0)[1]  # noqa: E731
dJdu0 = jax.grad(eval_J)(u0)

scaled_dJdu0_fenics = numpy_to_fenics(20 * dJdu0, fn.Function(V))

fn.plot(final_solution, label=f"u at t = {tspan[1]}")
fn.plot(u0_fenics, label=f"u at t = {tspan[0]}")
fn.plot(scaled_dJdu0_fenics, label=f"scaled x20 dJdu0")
plt.legend()
plt.show()
