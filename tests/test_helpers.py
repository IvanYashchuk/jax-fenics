import pytest

import fenics
import numpy
from jaxfenics import fenics_to_numpy, numpy_to_fenics


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (fenics.Constant(0.66), numpy.asarray(0.66)),
        (fenics.Constant([0.5, 0.66]), numpy.asarray([0.5, 0.66])),
    ],
)
def test_fenics_to_numpy_constant(test_input, expected):
    assert numpy.allclose(fenics_to_numpy(test_input), expected)


def test_fenics_to_numpy_function():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, "DG", 0)
    test_input = fenics.interpolate(fenics.Expression("x[0]", degree=1), V)
    expected = numpy.linspace(0.05, 0.95, num=10)
    assert numpy.allclose(fenics_to_numpy(test_input), expected)


def test_fenics_to_numpy_mixed_function():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = fenics.UnitIntervalMesh(10)
    vec_dim = 4
    V = fenics.VectorFunctionSpace(mesh, "DG", 0, dim=vec_dim)
    test_input = fenics.interpolate(
        fenics.Expression(vec_dim * ("x[0]",), element=V.ufl_element()), V
    )
    expected = numpy.linspace(0.05, 0.95, num=10)
    expected = numpy.tile(expected, (4, 1)).T
    assert numpy.allclose(fenics_to_numpy(test_input), expected)


def test_fenics_to_numpy_vector():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, "DG", 0)
    test_input = fenics.interpolate(fenics.Expression("x[0]", degree=1), V)
    test_input_vector = test_input.vector()
    expected = numpy.linspace(0.05, 0.95, num=10)
    assert numpy.allclose(fenics_to_numpy(test_input_vector), expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (numpy.asarray(0.66), fenics.Constant(0.66)),
        (numpy.asarray([0.5, 0.66]), fenics.Constant([0.5, 0.66])),
    ],
)
def test_numpy_to_fenics_constant(test_input, expected):
    fenics_test_input = numpy_to_fenics(test_input, fenics.Constant(0.0))
    assert numpy.allclose(fenics_test_input.values(), expected.values())


def test_numpy_to_fenics_function():
    test_input = numpy.linspace(0.05, 0.95, num=10)
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, "DG", 0)
    template = fenics.Function(V)
    fenics_test_input = numpy_to_fenics(test_input, template)
    expected = fenics.interpolate(fenics.Expression("x[0]", degree=1), V)
    assert numpy.allclose(
        fenics_test_input.vector().get_local(), expected.vector().get_local()
    )
