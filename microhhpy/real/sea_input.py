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
from microhhpy.interp import extrapolate_onto_mask
from microhhpy.interp import interp_rect_to_curv_latlon_2d


def create_sst_from_regular_latlon(
    sst_in,
    lon_in,
    lat_in,
    lon_out,
    lat_out,
    extrapolate_sea=True,
    float_type=np.float64):
    """
    Create sea surface temperature from regular lat/lon input.

    Based on the mask of `sst_in`, the known SST values are first
    extrapolated a few grid points onto land. Otherwise .. TODO

    Parameters:
    ----------
    sst_in : np.ma.masked_array, shape=(lon, lat) or (time, lon, lat)
        Input sea surface temperature. Mask = True indicates land points.
    lon_in : np.ndarray, shape=(lon,)
        Input longitude.
    lat_in : np.ndarray, shape=(lat,)
        Input latitude.
    lon_out : np.ndarray, shape=(lon_out, lat_out)
        Output longitude.
    lat_out : np.ndarray, shape=(lon_out, lat_out)
        Output latitude.
    extrapolate_sea : bool
        Extrapolate SST values onto land mask before interpolating.
    float_type : np.float32 or np.float64
        Floating point precision.

    Returns:
    -------
    sst_out : np.ndarray, shape=(lon_out, lat_out) or (time, lon_out, lat_out)
        Processed and interpolated SSTs.
    """

    if not isinstance(sst_in, np.ma.masked_array):
        logger.critical('Input SST has to be a masked array!')

    if isinstance(lon_in, np.ma.masked_array):
        lon_in = lon_in.data
    if isinstance(lat_in, np.ma.masked_array):
        lat_in = lat_in.data

    def process_tstep(sst_in):
        if extrapolate_sea:
            # Extrapolate function expects the sea mask.
            sea_mask = ~sst_in.mask

            # Extrapolate SSTs onto land-mask before interpolation.
            sst_ext = extrapolate_onto_mask(sst_in.data, sea_mask, max_distance=5)
        else:
            sst_ext = sst_in.data

        # Interpolate onto LES grid.
        return interp_rect_to_curv_latlon_2d(
                sst_ext, lon_in, lat_in, lon_out, lat_out, float_type=float_type)

    if sst_in.ndim == 3:
        ntime = sst_in.shape[0]
        sst_out = np.tile(lon_out, (ntime, 1, 1))

        for t in range(ntime):
            sst_out[t,:,:] = process_tstep(sst_in[t,:,:])
    else:
        sst_out = process_tstep(sst_in)

    if np.any(sst_out < 273.15):
        logger.warning('Interpolated/processed SSTs contain temperatures below zero!')

    return sst_out