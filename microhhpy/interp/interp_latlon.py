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
from .interp_kernels import Rect_to_curv_interpolation_factors
from .interp_kernels import interp_rect_to_curv_kernel


def interp_rect_to_curv_latlon_2d(
    fld_in,
    lon_in,
    lat_in,
    lon_out,
    lat_out,
    float_type):
    """
    Interpolate 2D `fld_in` from rectilinear grid to curvilinear grid.

    Arguments:
    ----------
    fld_in : np.ndarray, shape(2,) 
        Input field.
    lon_in, np.ndarray, shape(1,)
        Input longitudes.
    lat_in, np.ndarray, shape(1,)
        Input latitudes.
    lon_out, np.ndarray, shape(2,)
        Output longitudes.
    lat_out, np.ndarray, shape(2,)
        Output latitudes.
    float_type : np.float32 or np.float64
        Floating point precision output field.

    Returns:
    --------
    fld_out : np.ndarray, shape(2,)
        Interpolated field.
    """

    fld_out = np.zeros_like(lon_out, dtype=float_type)

    ipf = Rect_to_curv_interpolation_factors(lon_in, lat_in, lon_out, lat_out, float_type)
    interp_rect_to_curv_kernel(fld_out, fld_in, ipf.il, ipf.jl, ipf.fx, ipf.fy, z_out=None, z_in=None, float_type=float_type)

    return fld_out
