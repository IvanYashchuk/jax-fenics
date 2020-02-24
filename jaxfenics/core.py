import fenics

import jax
import jax.numpy as np

from jax.core import Primitive
from jax.interpreters.ad import defvjp, defvjp_all
from jax.api import defjvp_all

from .helpers import numpy_to_fenics, fenics_to_numpy


def build_fem_eval(ofunc, fenics_templates):
    """Return `f(*args) = build_fem_eval(ofunc(*args), args)`.
    Given the FEniCS-side function ofunc(*args), return the function
    `f(*args) = build_fem_eval(ofunc(*args), args)` with
    the VJP of `f` defined using `vjp_odeint`, where:
    `*args` are all arguments to `ofunc`.
    Args:
    ofunc: The FEniCS-side function to be wrapped.
    Returns:
    `f(args) = build_fem_eval(ofunc(*args), args)`
    """
    raise NotImplementedError
