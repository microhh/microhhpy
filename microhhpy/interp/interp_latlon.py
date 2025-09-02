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
from scipy.interpolate import RegularGridInterpolator
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
    float_type,
    method='linear'):
    """
    Wrapper for RegularGridInterpolator.

    Parameters:
    ----------
    fld_in : np.ndarray, ndim=2 
        Input field.
    lon_in, np.ndarray, ndim=1
        Input longitudes.
    lat_in, np.ndarray, ndim=1
        Input latitudes.
    lon_out, np.ndarray, ndim=2
        Output longitudes.
    lat_out, np.ndarray, ndim=2
        Output latitudes.
    float_type : np.float32 or np.float64
        Floating point precision output field.
    method : str
        Interpolation method.

    Returns:
    -------
    fld_out : np.ndarray, shape(2,)
        Interpolated field.
    """

    interp_func = RegularGridInterpolator(
        (lat_in, lon_in), fld_in.astype(float_type), method=method, bounds_error=True)

    points_out = np.column_stack([lat_out.ravel(), lon_out.ravel()])
    fld_out = interp_func(points_out).astype(float_type).reshape(lon_out.shape)

    return fld_out