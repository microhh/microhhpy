# Expose sub-directories as `import microhhpy; microhhpy.subdir.some_function()`
# NOTE: this only exposes what is defined in the subdirectory `__init__.py`.
# TODO: don't import *, Ruff complains about that.

from .chem import *
from .interp import *
from .io import *
from .land import *
from .solvers import *
from .spatial import *
from .rad import *
from .real import *
from .thermo import *
