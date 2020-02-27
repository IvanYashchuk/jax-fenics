# jax-fenics &middot; [![Build](https://github.com/ivanyashchuk/jax-fenics/workflows/CI/badge.svg)](https://github.com/ivanyashchuk/jax-fenics/actions?query=workflow%3ACI+branch%3Amaster) [![Coverage Status](https://coveralls.io/repos/github/IvanYashchuk/jax-fenics/badge.svg?branch=master)](https://coveralls.io/github/IvanYashchuk/jax-fenics?branch=master)

This package enables use of FEniCS for solving differentiable variational problems in JAX.

## Installation
First install [FEniCS](http://fenicsproject.org).
Then install [JAX](https://github.com/google/jax) with:

    python -m pip install pip install --upgrade jax jaxlib  # CPU-only version

After that install the jax-fenics with:

    python -m pip install git+https://github.com/IvanYashchuk/jax-fenics.git@master

## Reporting bugs

If you found a bug, create an [issue].

[issue]: https://github.com/IvanYashchuk/jax-fenics/issues/new

## Contributing

Pull requests are welcome from everyone.

Fork, then clone the repository:

    git clone https://github.com/IvanYashchuk/jax-fenics.git

Make your change. Add tests for your change. Make the tests pass:

    pytest tests/

Check the formatting with `black` and `flake8`. Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/IvanYashchuk/jax-fenics/pulls