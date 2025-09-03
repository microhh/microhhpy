# MicroHHpy

---
[![PyPi Badge](https://img.shields.io/pypi/v/microhhpy.svg?colorB=blue)](https://pypi.python.org/pypi/microhhpy/)
---

Python package with utility functions for working with MicroHH LES/DNS.

> [!IMPORTANT]  
> This Python package is available on PyPI (https://pypi.org/project/microhhpy/), but mostly as a placeholder. Since `microhhpy` is actively developing and unstable, the PyPI version may be outdated.

### Usage
Either add the `microhhpy` package location to your `PYTHONPATH`:

    export PYTHONPATH="${PYTHONPATH}:/path/to/microhhpy"

Or specify the path using `sys`, before importing `microhhpy`:

    import sys
    sys.path.append('/path/to/microhhpy')

Now `microhhpy` should be available as an import, e.g.:

    from microhhpy.spatial import Domain
    from microhhpy.spatial import Projection
