# MicroHHpy

Python package with utility functions for working with MicroHH LES/DNS.

### Usage
Either add the `microhhpy` package location to your `PYTHONPATH`:

    export PYTHONPATH="${PYTHONPATH}:/path/to/microhhpy"

Or specify the path using `sys`, before importing `microhhpy`:

    import sys
    sys.path.append('/path/to/microhhpy')

Now `microhhpy` should be available as an import, e.g.:

    from microhhpy.spatial import Domain
    from microhhpy.spatial import Projection
