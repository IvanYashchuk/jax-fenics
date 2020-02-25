import fenics
import jax
import jax.numpy as np


def fenics_to_numpy(fenics_var):
    """Convert FEniCS variable to numpy/jax array.
    Serializes the input so that all processes have the same data."""
    if isinstance(fenics_var, fenics.Constant):
        return np.asarray(fenics_var.values())

    if isinstance(fenics_var, fenics.Function):
        fenics_vec = fenics_var.vector()
        data = fenics_vec.gather(np.arange(fenics_vec.size(), dtype="I"))
        n_sub = fenics_var.function_space().num_sub_spaces()
        # Reshape if function is in vector function space
        if n_sub != 0:
            data = np.reshape(data, (len(data) // n_sub, n_sub))
        return np.asarray(data)

    if isinstance(fenics_var, fenics.GenericVector):
        return np.asarray(
            fenics_var.gather(np.arange(fenics_var.size(), dtype="I"))
        )

    raise ValueError("Cannot convert " + str(type(fenics_var)))


def numpy_to_fenics(numpy_array, fenics_var_template):
    """Convert numpy/jax array to FEniCS variable"""

    if isinstance(fenics_var_template, fenics.Constant):

        # JAX tracer specific part. Here we return zero values if tracer is not ConcreteArray type.
        if isinstance(numpy_array, (jax.core.Tracer,)):
            numpy_array = jax.core.get_aval(numpy_array)

        if isinstance(numpy_array, (jax.abstract_arrays.ShapedArray,)):
            if not isinstance(
                numpy_array, (jax.abstract_arrays.ConcreteArray,)
            ):
                import warnings

                warnings.warn(
                    "Got JAX tracer type to convert to FEniCS. Returning zero."
                )
                if numpy_array.shape == (1,):
                    return type(fenics_var_template)(0.0)
                else:
                    return type(fenics_var_template)(
                        np.zeros_like(fenics_var_template.values())
                    )

        if isinstance(numpy_array, (jax.abstract_arrays.ConcreteArray,)):
            numpy_array = numpy_array.val

        if isinstance(numpy_array, jax.ad_util.Zero):
            numpy_array = np.zeros_like(fenics_var_template.values())

        if numpy_array.shape == (1,):
            return type(fenics_var_template)(numpy_array[0])
        else:
            return type(fenics_var_template)(numpy_array)

    if isinstance(fenics_var_template, fenics.Function):
        np_n_sub = numpy_array.shape[-1]
        np_size = numpy_array.size

        function_space = fenics_var_template.function_space()

        u = type(fenics_var_template)(function_space)

        if isinstance(numpy_array, jax.ad_util.Zero):
            return u

        # assume that given numpy/jax array is global array that needs to be distrubuted across processes
        # when FEniCS function is created
        fenics_size = u.vector().size()
        fenics_n_sub = function_space.num_sub_spaces()

        if (
            fenics_n_sub != 0 and np_n_sub != fenics_n_sub
        ) or np_size != fenics_size:
            err_msg = (
                f"Cannot convert numpy array to Function:"
                f"Wrong size {numpy_array.size} vs {u.vector().size()}"
            )
            raise ValueError(err_msg)

        if numpy_array.dtype != np.float_:
            err_msg = (
                f"The numpy array must be of type {np.float_}, "
                "but got {numpy_array.dtype}"
            )
            raise ValueError(err_msg)

        if isinstance(numpy_array, (jax.core.Tracer,)):
            numpy_array = jax.core.get_aval(numpy_array)

        if isinstance(numpy_array, (jax.abstract_arrays.ShapedArray,)):
            if not isinstance(
                numpy_array, (jax.abstract_arrays.ConcreteArray,)
            ):
                import warnings

                warnings.warn(
                    "Got JAX tracer type to convert to FEniCS. Returning zero."
                )
                return u

        if isinstance(numpy_array, (jax.abstract_arrays.ConcreteArray,)):
            numpy_array = numpy_array.val

        range_begin, range_end = u.vector().local_range()
        local_array = numpy_array.reshape(fenics_size)[range_begin:range_end]
        u.vector().set_local(local_array)
        u.vector().apply("insert")
        return u

    err_msg = f"Cannot convert numpy/jax array to {fenics_var_template}"
    raise ValueError(err_msg)
