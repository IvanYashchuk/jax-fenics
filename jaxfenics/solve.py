import fenics
import ufl

import jax
import jax.numpy as np

from jax.core import Primitive
from jax.interpreters.ad import defvjp, defvjp_all
from jax.api import defjvp_all

import functools

from .helpers import (
    numpy_to_fenics,
    fenics_to_numpy,
    get_numpy_input_templates,
    check_input,
    convert_all_to_fenics,
)
from .helpers import FenicsVariable

from typing import Type, List, Union, Iterable, Callable, Tuple


def solve_eval(
    fenics_function: Callable,
    fenics_templates: Iterable[FenicsVariable],
    *args: np.array,
) -> Tuple[np.array, ufl.Form, Tuple[FenicsVariable], List[fenics.DirichletBC]]:
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
            "FEniCS function output should be in the form (solution, residual_form, [bcs])."
        )

    fenics_solution, residual_form, bcs = out

    if isinstance(fenics_solution, tuple):
        raise ValueError(
            "Only single solution output from FEniCS function is supported."
        )

    if not isinstance(fenics_solution, fenics.Function):
        raise ValueError(
            f"FEniCS function output should be in the form (solution, residual_form, [bcs]). Got {type(fenics_solution)} instead of fenics.Function"
        )

    if not isinstance(residual_form, ufl.Form):
        raise ValueError(
            f"FEniCS function output should be in the form (solution, residual_form, [bcs]). Got {type(residual_form)} instead of ufl.Form"
        )

    if not isinstance(bcs, list):
        raise ValueError(
            f"FEniCS function output should be in the form (solution, residual_form, [bcs]). Got {bcs}."
        )

    numpy_output = np.asarray(fenics_to_numpy(fenics_solution))
    return numpy_output, fenics_solution, residual_form, fenics_inputs, bcs


def vjp_solve_eval(
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

    numpy_output, fenics_solution, residual_form, fenics_inputs, bcs = solve_eval(
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
            for i, vjp in enumerate(
                vjp_solve_eval_impl(
                    g, fenics_solution, residual_form, fenics_inputs, bcs
                )
            )
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
        """Computes the batched version of the primitive.

        This must be a JAX-traceable function.

        Since the vjp_fun1 primitive already operates pointwise on arbitrary
        dimension tensors, to batch it we can use the primitive itself. This works as
        long as both the inputs have the same dimensions and are batched along the
        same axes. The result is batched along the axis that the inputs are batched.

        Args:
            vector_arg_values: a tuple of two arguments, each being a tensor of matching
            shape.
            batch_axes: the axes that are being batched. See vmap documentation.
        Returns:
            a tuple of the result, and the result axis that was batched.
        """
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


def _action(A, x):
    A = ufl.algorithms.expand_derivatives(A)
    if A.integrals() != ():  # form is not empty:
        return ufl.action(A, x)
    else:
        return A  # form is empty, doesn't matter


# @trace("vjp_solve_eval_impl")
def vjp_solve_eval_impl(
    g: np.array,
    fenics_solution: fenics.Function,
    fenics_residual: ufl.Form,
    fenics_inputs: List[FenicsVariable],
    bcs: List[fenics.DirichletBC],
) -> Tuple[np.array]:
    """Computes the gradients of the output with respect to the inputs."""
    # Convert tangent covector (adjoint) to a FEniCS variable
    adj_value = numpy_to_fenics(g, fenics_solution)
    adj_value = adj_value.vector()

    F = fenics_residual
    u = fenics_solution
    V = u.function_space()
    dFdu = fenics.derivative(F, u)
    adFdu = ufl.adjoint(
        dFdu, reordered_arguments=ufl.algorithms.extract_arguments(dFdu)
    )

    u_adj = fenics.Function(V)
    adj_F = ufl.action(adFdu, u_adj)
    adj_F = ufl.replace(adj_F, {u_adj: fenics.TrialFunction(V)})
    adj_F_assembled = fenics.assemble(adj_F)

    for bc in bcs:
        bc.homogenize()
    hbcs = bcs

    for bc in hbcs:
        bc.apply(adj_F_assembled)
        bc.apply(adj_value)

    fenics.solve(adj_F_assembled, u_adj.vector(), adj_value)

    fenics_grads = []
    for fenics_input in fenics_inputs:
        if isinstance(fenics_input, fenics.Function):
            V = fenics_input.function_space()
        dFdm = fenics.derivative(F, fenics_input, fenics.TrialFunction(V))
        adFdm = fenics.adjoint(dFdm)
        result = fenics.assemble(-adFdm * u_adj)
        if isinstance(fenics_input, fenics.Constant):
            fenics_grad = fenics.Constant(result.sum())
        else:  # fenics.Function
            fenics_grad = fenics.Function(V, result)
        fenics_grads.append(fenics_grad)

    # Convert FEniCS gradients to jax array representation
    jax_grads = (
        None if fg is None else np.asarray(fenics_to_numpy(fg)) for fg in fenics_grads
    )

    jax_grad_tuple = tuple(jax_grads)

    return jax_grad_tuple


def jvp_solve_eval(
    fenics_function: Callable,
    fenics_templates: Iterable[FenicsVariable],
    primals: Tuple[np.array],
    tangents: Tuple[np.array],
) -> Tuple[np.array]:
    """Computes the tangent linear model
    """

    (
        numpy_output_primal,
        fenics_solution_primal,
        residual_form,
        fenics_primals,
        bcs,
    ) = solve_eval(fenics_function, fenics_templates, *primals)

    # Now tangent evaluation!
    F = residual_form
    u = fenics_solution_primal
    V = u.function_space()

    for bc in bcs:
        bc.homogenize()
    hbcs = bcs

    fenics_tangents = convert_all_to_fenics(fenics_primals, *tangents)
    fenics_output_tangents = []
    for fp, ft in zip(fenics_primals, fenics_tangents):
        dFdu = fenics.derivative(F, u)
        dFdm = fenics.derivative(F, fp, ft)
        u_tlm = fenics.Function(V)
        tlm_F = ufl.action(dFdu, u_tlm) + dFdm
        tlm_F = ufl.replace(tlm_F, {u_tlm: fenics.TrialFunction(V)})
        fenics.solve(ufl.lhs(tlm_F) == ufl.rhs(tlm_F), u_tlm, bcs=hbcs)
        fenics_output_tangents.append(u_tlm)

    jax_output_tangents = (fenics_to_numpy(ft) for ft in fenics_output_tangents)
    jax_output_tangent = sum(jax_output_tangents)

    return numpy_output_primal, jax_output_tangent


def build_jax_solve_eval(fenics_templates: FenicsVariable) -> Callable:
    """Return `f(*args) = build_jax_solve_eval(*args)(ofunc(*args))`.
    Given the FEniCS-side function ofunc(*args), return the function
    `f(*args) = build_jax_solve_eval(*args)(ofunc(*args))` with
    the VJP of `f`, where:
    `*args` are all arguments to `ofunc`.
    Args:
    ofunc: The FEniCS-side function to be wrapped.
    Returns:
    `f(args) = build_jax_solve_eval(*args)(ofunc(*args))`
    """

    def decorator(fenics_function: Callable) -> Callable:
        @functools.wraps(fenics_function)
        def jax_solve_eval(*args):
            return jax_solve_eval_p.bind(*args)

        jax_solve_eval_p = Primitive("jax_solve_eval")
        jax_solve_eval_p.def_impl(
            lambda *args: solve_eval(fenics_function, fenics_templates, *args)[0]
        )

        jax_solve_eval_p.def_abstract_eval(
            lambda *args: jax.abstract_arrays.make_shaped_array(
                solve_eval(fenics_function, fenics_templates, *args)[0]
            )
        )

        def jax_solve_eval_batch(vector_arg_values, batch_axes):
            assert len(set(batch_axes)) == 1  # assert that all batch axes are same
            assert (
                batch_axes[0] == 0
            )  # assert that batch axis is zero, need to rewrite for a general case?
            # compute function row-by-row
            res = np.asarray(
                [
                    jax_solve_eval(
                        *(vector_arg_values[j][i] for j in range(len(batch_axes)))
                    )
                    for i in range(vector_arg_values[0].shape[0])
                ]
            )
            return res, batch_axes[0]

        jax.batching.primitive_batchers[jax_solve_eval_p] = jax_solve_eval_batch

        # @trace("djax_solve_eval")
        def djax_solve_eval(*args):
            return djax_solve_eval_p.bind(*args)

        djax_solve_eval_p = Primitive("djax_solve_eval")
        # djax_solve_eval_p.multiple_results = True
        djax_solve_eval_p.def_impl(
            lambda *args: vjp_solve_eval(fenics_function, fenics_templates, *args)
        )

        defvjp_all(jax_solve_eval_p, djax_solve_eval)
        return jax_solve_eval

    return decorator


# it seems that it is not possible to define custom vjp and jvp rules simultaneusly
# at least I did not figure out how to do this
# they override each other
# therefore here I create a separate wrapped function
def build_jax_solve_eval_fwd(fenics_templates: FenicsVariable) -> Callable:
    """Return `f(*args) = build_jax_solve_eval(*args)(ofunc(*args))`. This is forward mode AD.
    Given the FEniCS-side function ofunc(*args), return the function
    `f(*args) = build_jax_solve_eval(*args)(ofunc(*args))` with
    the JVP of `f`, where:
    `*args` are all arguments to `ofunc`.
    Args:
    ofunc: The FEniCS-side function to be wrapped.
    Returns:
    `f(args) = build_jax_solve_eval(*args)(ofunc(*args))`
    """

    def decorator(fenics_function: Callable) -> Callable:
        @functools.wraps(fenics_function)
        def jax_solve_eval(*args):
            return jax_solve_eval_p.bind(*args)

        jax_solve_eval_p = Primitive("jax_solve_eval")
        jax_solve_eval_p.def_impl(
            lambda *args: solve_eval(fenics_function, fenics_templates, *args)[0]
        )

        jax_solve_eval_p.def_abstract_eval(
            lambda *args: jax.abstract_arrays.make_shaped_array(
                solve_eval(fenics_function, fenics_templates, *args)[0]
            )
        )

        def jax_solve_eval_batch(vector_arg_values, batch_axes):
            assert len(set(batch_axes)) == 1  # assert that all batch axes are same
            assert (
                batch_axes[0] == 0
            )  # assert that batch axis is zero, need to rewrite for a general case?
            # compute function row-by-row
            res = np.asarray(
                [
                    jax_solve_eval(
                        *(vector_arg_values[j][i] for j in range(len(batch_axes)))
                    )
                    for i in range(vector_arg_values[0].shape[0])
                ]
            )
            return res, batch_axes[0]

        jax.batching.primitive_batchers[jax_solve_eval_p] = jax_solve_eval_batch

        # @trace("jvp_jax_solve_eval")
        def jvp_jax_solve_eval(ps, ts):
            return jvp_jax_solve_eval_p.bind(ps, ts)

        jvp_jax_solve_eval_p = Primitive("jvp_jax_solve_eval")
        jvp_jax_solve_eval_p.multiple_results = True
        jvp_jax_solve_eval_p.def_impl(
            lambda ps, ts: jvp_solve_eval(fenics_function, fenics_templates, ps, ts)
        )

        jax.interpreters.ad.primitive_jvps[jax_solve_eval_p] = jvp_jax_solve_eval

        # TODO: JAX Tracer goes inside fenics wrappers and zero array is returned
        # because fenics numpy conversion works only for concrete arrays
        vjp_jax_solve_eval_p = Primitive("vjp_jax_solve_eval")
        vjp_jax_solve_eval_p.def_impl(
            lambda ct, *args: vjp_solve_eval(fenics_function, fenics_templates, *args)[
                1
            ](ct)
        )

        jax.interpreters.ad.primitive_transposes[
            jax_solve_eval_p
        ] = vjp_jax_solve_eval_p

        return jax_solve_eval

    return decorator
