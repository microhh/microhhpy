#
#  MicroHH
#  Copyright (c) 2011-2024 Chiel van Heerwaarden
#  Copyright (c) 2011-2024 Thijs Heus
#  Copyright (c) 2014-2024 Bart van Stratum
#
#  This file is part of MicroHH
#
#  MicroHH is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  MicroHH is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
#

# Standard library

# Third-party
import numpy as np

# Local library  

"""
Use `snake_case` case for variable and function names.
"""
some_variable = 1.0

def some_function():
    return None


"""
Classes start with a capital letter, instances without.
"""
class Some_class:
    def __init__(self):
        self.something = None

my_class = Some_class()


"""
Avoid classes to store simple properties.
"""
class Calc_something:
    def __init__(self, a, b):
        self.a = 2*a
        self.b = 2*b

# Instead:
def calc_something(a, b):
    a = 2*a
    b = 2*b
    return a, b

# Or:
def calc_something(a, b):
    return dict(
        a=2*a,
        b=2*b)


"""
Use the logger for output messages.
"""
from microhhpy.logger import logger

logger.debug("Debug or very detailled output")
logger.info("Info on workflow")
logger.warning("Non-critical warnings")
logger.error("Non-critical errors")
logger.critical("Critical error - raises RuntimeError()")


"""
Organize imports in three groups.
"""
# Standard library
import datetime
import os

# Third-party
import numpy as np
import pandas as pd

# Local library  
import microhhpy.constants as cst


"""
Properly document functions/classes.
 - Indicate data type and/or dimensions where necessary/critical.
 - Use `float_type` for floating point precision, not `dtype`.
"""
def function(array_1, array_2, some_number, float_type=np.float64, optional_arg='woef'):
    """
    Function description.

    Parameters:
    ----------
    array_1 : np.ndarray(dtype=float, ndim=2)
        Some array (units).
    array_2 : np.ndarray(dtype=int, ndim=1)
        Another array (units).
    some_number : float
        A floating point number.
    float_type : np.float32 or np.float64, optional
        Floating point precision, default=np.float64
    optional_arg : str, optional
        An argument, default='woef'

    Returns:
    -------
    array_3 : np.ndarray(dtype=float, ndim=2)
        Output array.
    """
    array_3 = 2*array_1
    return array_3