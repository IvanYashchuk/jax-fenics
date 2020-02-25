import fenics
import ufl

import jax
import jax.numpy as np

from jax.core import Primitive
from jax.interpreters.ad import defvjp, defvjp_all
from jax.api import defjvp_all

from .helpers import numpy_to_fenics, fenics_to_numpy

from typing import Type, List, Union, Iterable, Callable, Tuple

FenicsVariable = Union[fenics.Constant, fenics.Function]


def get_numpy_input_templates(
    fenics_input_templates: Iterable[FenicsVariable],
) -> List[np.array]:
    """Returns tuple of numpy representations of the input templates to FEniCSFunctional.forward"""
    numpy_input_templates = [fenics_to_numpy(x) for x in fenics_input_templates]
    return numpy_input_templates


def check_input(fenics_templates: FenicsVariable, *args: FenicsVariable) -> None:
    """Checks that the number of inputs arguments is correct"""
    n_args = len(args)
    expected_nargs = len(fenics_templates)
    if n_args != expected_nargs:
        raise ValueError(
            "Wrong number of arguments"
            " Expected {} got {}.".format(expected_nargs, n_args)
        )

    # Check that each input argument has correct dimensions
    numpy_templates = get_numpy_input_templates(fenics_templates)
    for i, (arg, template) in enumerate(zip(args, numpy_templates)):
        if arg.shape != template.shape:
            raise ValueError(
                "Expected input shape {} for input"
                " {} but got {}.".format(template.shape, i, arg.shape)
            )

    # Check that the inputs are of double precision
    for i, arg in enumerate(args):
        if arg.dtype != np.float64:
            raise TypeError(
                "All inputs must be type {},"
                " but got {} for input {}.".format(np.float64, arg.dtype, i)
            )


def convert_all_to_fenics(
    fenics_templates: FenicsVariable, *args: np.array
) -> List[FenicsVariable]:
    """Converts input array to corresponding FEniCS variables"""
    fenics_inputs = []
    for inp, template in zip(args, fenics_templates):
        fenics_inputs.append(numpy_to_fenics(inp, template))
    return fenics_inputs


def fem_eval(
    fenics_function: Callable,
    fenics_templates: Iterable[FenicsVariable],
    *args: FenicsVariable,
) -> Tuple[np.array, ufl.Form, Tuple[FenicsVariable]]:
    """Computes the output of a fenics_function and saves a corresponding gradient tape
    Input:
        fenics_function (callable): FEniCS function to be executed during the forward pass
        args (tuple): jax array representation of the input to fenics_function
    Output:
        numpy_output (np.array): JAX array representation of the output from fenics_function(*fenics_inputs)
        F_form (ufl.Form): UFL Form for the residual used to solve the problem with fenics.solve(F==0, ...)
        fenics_inputs (list of FenicsVariable): FEniCS representation of the input args
    """

    check_input(fenics_templates, *args)
    fenics_inputs = convert_all_to_fenics(fenics_templates, *args)

    fenics_solution, F_form = fenics_function(*fenics_inputs)

    if isinstance(fenics_solution, tuple):
        raise ValueError(
            "Only single solution output from FEniCS function is supported."
        )

    if not isinstance(fenics_solution, fenics.Function):
        raise ValueError(
            f"FEniCS function output should be in the form (solution, F_form). Got {type(fenics_solution)} instead of fenics.Function"
        )

    if not isinstance(F_form, ufl.Form):
        raise ValueError(
            f"FEniCS function output should be in the form (solution, F_form). Got {type(F_form)} instead of ufl.Form"
        )

    numpy_output = np.asarray(fenics_to_numpy(fenics_solution))
    return numpy_output, F_form, fenics_inputs


def build_fem_eval(ofunc: Callable, fenics_templates: FenicsVariable) -> Callable:
    """Return `f(*args) = build_fem_eval(ofunc(*args), args)`.
    Given the FEniCS-side function ofunc(*args), return the function
    `f(*args) = build_fem_eval(ofunc(*args), args)` with
    the VJP of `f`, where:
    `*args` are all arguments to `ofunc`.
    Args:
    ofunc: The FEniCS-side function to be wrapped.
    Returns:
    `f(args) = build_fem_eval(ofunc(*args), args)`
    """
    raise NotImplementedError
