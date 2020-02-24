from distutils.core import setup

setup(
    name="jaxfenics",
    description="JAX-FEniCS interface",
    url="https://github.com/IvanYashchuk/jax-fenics",
    author="Ivan Yashchuk",
    license="MIT",
    packages=["jaxfenics"],
    install_requires=["jax", "fenics"],
)
