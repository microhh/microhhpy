# Expose some main function as `microhhpy.some_function()` instead of having to use `microhhpy.subdir.some_function()`.
#from .main.initial_fields import create_initial_fields

# Expose sub-directories as `import microhhpy; microhhpy.subdir.some_function()`
# NOTE: this only exposes what is defined in the subdirectory `__init__.py`.
from .chem import *
from .interp import *
from .io import *
from .land import *
from .spatial import *
from .real import *
from .thermo import *
