# MicroHHpy

---
[![PyPi Badge](https://img.shields.io/pypi/v/microhhpy.svg?colorB=blue)](https://pypi.python.org/pypi/microhhpy/)
---

### Description

Python package with utility functions for working with MicroHH LES/DNS.

The main purpose of `microhhpy` is to simplify complex workflows and case setups, such as setting up nested simulations with open boundary conditions. Basic scripts for handling MicroHH input and output are still available in `microhh/python/microhh_tools.py` and other scripts like `cross_to_nc.py`.

> [!IMPORTANT]  
> Like `microhh` itself, this Python package has only been tested in Unix environments.

### Usage

The recommended way to install `microhhpy` is via PyPI:

    pip install microhhpy

For development, you can clone the repository and make `microhhpy` available in one of two ways:

1. Add it to your `PYTHONPATH`:

       export PYTHONPATH="${PYTHONPATH}:/path/to/microhhpy"

2. Or append the path directly in your Python script before importing:

       import sys
       sys.path.append('/path/to/microhhpy')

Once set up, you can import `microhhpy` modules, for example:

    from microhhpy.spatial import Domain
    from microhhpy.spatial import Projection
