import fenics
import ufl

import jax
import jax.numpy as np

from jax.core import Primitive
from jax.interpreters.ad import defvjp, defvjp_all
from jax.api import defjvp_all

from .helpers import (
    numpy_to_fenics,
    fenics_to_numpy,
    get_numpy_input_templates,
    check_input,
    convert_all_to_fenics,
)
from .helpers import FenicsVariable

from typing import Type, List, Union, Iterable, Callable, Tuple


def assemble_eval(
    fenics_function: Callable,
    fenics_templates: Iterable[FenicsVariable],
    *args: np.array,
) -> Tuple[np.array, ufl.Form, Tuple[FenicsVariable]]:
    """Computes the output of a fenics_function and saves a corresponding gradient tape
    Input:
        fenics_function (callable): FEniCS function to be executed during the forward pass
        fenics_templates (iterable of FenicsVariable): Templates for converting arrays to FEniCS types
        args (tuple): jax array representation of the input to fenics_function
    Output:
        numpy_output (np.array): JAX array representation of the output from fenics_function(*fenics_inputs)
        residual_form (ufl.Form): UFL Form for the residual used to solve the problem with fenics.solve(F==0, ...)
        fenics_inputs (list of FenicsVariable): FEniCS representation of the input args
    """

    check_input(fenics_templates, *args)
    fenics_inputs = convert_all_to_fenics(fenics_templates, *args)

    out = fenics_function(*fenics_inputs)
    if not isinstance(out, tuple):
        raise ValueError(
            "FEniCS function output should be in the form (assembly_output, ufl_form)."
        )

    assembly_output, ufl_form = out

    if isinstance(assembly_output, tuple):
        raise ValueError(
            "Only single solution output from FEniCS function is supported."
        )

    if not isinstance(assembly_output, float):
        raise ValueError(
            f"FEniCS function output should be in the form (assembly_output, ufl_form). Got {type(assembly_output)} instead of float"
        )

    if not isinstance(ufl_form, ufl.Form):
        raise ValueError(
            f"FEniCS function output should be in the form (assembly_output, ufl_form). Got {type(ufl_form)} instead of ufl.Form"
        )

    numpy_output = np.asarray(assembly_output)
    return numpy_output, ufl_form, fenics_inputs


def vjp_assemble_eval(
    fenics_function: Callable, fenics_templates: FenicsVariable, *args: np.array
) -> Tuple[np.array, Callable]:
    """Computes the gradients of the output with respect to the input
    Input:
        fenics_function (callable): FEniCS function to be executed during the forward pass
        args (tuple): jax array representation of the input to fenics_function
    Output:
        A pair where the first element is the value of fun applied to the arguments and the second element
        is a Python callable representing the VJP map from output cotangents to input cotangents.
        The returned VJP function must accept a value with the same shape as the value of fun applied
        to the arguments and must return a tuple with length equal to the number of positional arguments to fun.
    """

    numpy_output, ufl_form, fenics_inputs = assemble_eval(
        fenics_function, fenics_templates, *args
    )

    # @trace("vjp_fun1")
    def vjp_fun1(g):
        return vjp_fun1_p.bind(g)

    vjp_fun1_p = Primitive("vjp_fun1")
    vjp_fun1_p.multiple_results = True
    vjp_fun1_p.def_impl(
        lambda g: tuple(
            vjp if vjp is not None else jax.ad_util.zeros_like_jaxval(args[i])
            for i, vjp in enumerate(vjp_assemble_impl(g, ufl_form, fenics_inputs))
        )
    )

    # @trace("vjp_fun1_abstract_eval")
    def vjp_fun1_abstract_eval(g):
        if len(args) > 1:
            return tuple(
                (jax.abstract_arrays.ShapedArray(arg.shape, arg.dtype) for arg in args)
            )
        else:
            return (
                jax.abstract_arrays.ShapedArray((1, *args[0].shape), args[0].dtype),
            )

    vjp_fun1_p.def_abstract_eval(vjp_fun1_abstract_eval)

    # @trace("vjp_fun1_batch")
    def vjp_fun1_batch(vector_arg_values, batch_axes):
        # _trace("Using vjp_fun1 to compute the batch:")
        assert (
            batch_axes[0] == 0
        )  # assert that batch axis is zero, need to rewrite for a general case?
        # compute function row-by-row
        res = np.asarray(
            [
                vjp_fun1(vector_arg_values[0][i])
                for i in range(vector_arg_values[0].shape[0])
            ]
        )
        return [res[:, i] for i in range(len(args))], (batch_axes[0],) * len(args)

    jax.batching.primitive_batchers[vjp_fun1_p] = vjp_fun1_batch

    return numpy_output, vjp_fun1


# @trace("vjp_assemble_impl")
def vjp_assemble_impl(
    g: np.array, fenics_output_form: ufl.Form, fenics_inputs: List[FenicsVariable],
) -> Tuple[np.array]:
    """Computes the gradients of the output with respect to the inputs."""

    # Compute derivative form for the output with respect to each input
    fenics_grads_forms = []
    for fenics_input in fenics_inputs:
        # Need to construct direction (test function) first
        if isinstance(fenics_input, fenics.Function):
            V = fenics_input.function_space()
        elif isinstance(fenics_input, fenics.Constant):
            mesh = fenics_output_form.ufl_domain().ufl_cargo()
            V = fenics.FunctionSpace(mesh, "Real", 0)
        else:
            raise NotImplementedError

        dv = fenics.TestFunction(V)
        fenics_grad_form = fenics.derivative(fenics_output_form, fenics_input, dv)
        fenics_grads_forms.append(fenics_grad_form)

    # Assemble the derivative forms
    fenics_grads = [fenics.assemble(form) for form in fenics_grads_forms]

    # Convert FEniCS gradients to jax array representation
    jax_grads = (
        None if fg is None else np.asarray(g * fenics_to_numpy(fg))
        for fg in fenics_grads
    )

    jax_grad_tuple = tuple(jax_grads)

    return jax_grad_tuple


def jvp_assemble_eval(
    fenics_function: Callable,
    fenics_templates: Iterable[FenicsVariable],
    primals: Tuple[np.array],
    tangents: Tuple[np.array],
) -> Tuple[np.array]:
    """Computes the jacobian-vector product for fenics.assemble
    """

    numpy_output_primal, output_primal_form, fenics_primals = assemble_eval(
        fenics_function, fenics_templates, *primals
    )

    # Now tangent evaluation!
    fenics_tangents = convert_all_to_fenics(fenics_primals, *tangents)
    output_tangent_form = 0.0
    for fp, ft in zip(fenics_primals, fenics_tangents):
        output_tangent_form += fenics.derivative(output_primal_form, fp, ft)

    if not isinstance(output_tangent_form, float):
        output_tangent_form = ufl.algorithms.expand_derivatives(output_tangent_form)
        output_tangent = fenics.assemble(output_tangent_form)

    jax_output_tangent = output_tangent

    return numpy_output_primal, jax_output_tangent


def build_jax_assemble_eval(
    ofunc: Callable, fenics_templates: FenicsVariable
) -> Callable:
    """Return `f(*args) = build_jax_assemble_eval(ofunc(*args), args)`.
    Given the FEniCS-side function ofunc(*args), return the function
    `f(*args) = build_jax_assemble_eval(ofunc(*args), args)` with
    the VJP of `f`, where:
    `*args` are all arguments to `ofunc`.
    Args:
    ofunc: The FEniCS-side function to be wrapped.
    Returns:
    `f(args) = build_jax_assemble_eval(ofunc(*args), args)`
    """

    def jax_assemble_eval(*args):
        return jax_assemble_eval_p.bind(*args)

    jax_assemble_eval_p = Primitive("jax_assemble_eval")
    jax_assemble_eval_p.def_impl(
        lambda *args: assemble_eval(ofunc, fenics_templates, *args)[0]
    )

    jax_assemble_eval_p.def_abstract_eval(
        lambda *args: jax.abstract_arrays.make_shaped_array(
            assemble_eval(ofunc, fenics_templates, *args)[0]
        )
    )

    def jax_assemble_eval_batch(vector_arg_values, batch_axes):
        assert len(set(batch_axes)) == 1  # assert that all batch axes are same
        assert (
            batch_axes[0] == 0
        )  # assert that batch axis is zero, need to rewrite for a general case?
        # compute function row-by-row
        res = np.asarray(
            [
                jax_assemble_eval(
                    *(vector_arg_values[j][i] for j in range(len(batch_axes)))
                )
                for i in range(vector_arg_values[0].shape[0])
            ]
        )
        return res, batch_axes[0]

    jax.batching.primitive_batchers[jax_assemble_eval_p] = jax_assemble_eval_batch

    # @trace("djax_assemble_eval")
    def djax_assemble_eval(*args):
        return djax_assemble_eval_p.bind(*args)

    djax_assemble_eval_p = Primitive("djax_assemble_eval")
    # djax_assemble_eval_p.multiple_results = True
    djax_assemble_eval_p.def_impl(
        lambda *args: vjp_assemble_eval(ofunc, fenics_templates, *args)
    )

    defvjp_all(jax_assemble_eval_p, djax_assemble_eval)

    return jax_assemble_eval
