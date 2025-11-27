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

# Third-party.
import numpy as np

# Local library
from microhhpy.logger import logger


def create_2d_coriolis_freq(lat, float_type):
    """
    Calculate 2D Coriolis frequency f = 2 Ω sin(lat)

    Parameters:
    ----------
    lat : np.ndarray, shape (jtot, itot)
        Input latitude field.

    Returns:
    -------
    fc : np.ndarray, shape (jtot, itot)
        2D Coriolis parameter.
    """
    T = 86400
    omega = 2 * np.pi / T

    fc = 2 * omega * np.sin(np.deg2rad(lat))
    fc = fc.astype(float_type)

    return fc