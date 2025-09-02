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
from scipy.interpolate import griddata
from scipy import ndimage

# Local library


def extrapolate_onto_mask(data, mask, max_distance=5):
    """
    Extrapolate 2D field onto masked area over a distance of `max_distance` grid points.
    A nearest-neighbour interpolation is used to ensure that no new extremes are created.
    
    Parameters:
    ----------
    data : np.ndarray, shape(j, i)
        Input field.
    mask : np.ndarray, bool, same shape as data
        Mask where True indicates valid values.
    max_distance : int
        Maximum distance in grid cells to extrapolate
    
    Returns:
    -------
    np.ndarray
        Data with values extrapolated onto nearby masked areas
    """
    
    source_data = data[mask]
    source_coords = np.column_stack(np.where(mask))
    
    # Calculate distance from each target cell to nearest source cell
    distance_to_source = ndimage.distance_transform_edt(~mask)
    
    # Only keep target cells within max_distance
    close_to_source = distance_to_source <= max_distance
    
    # Get coordinates of target cells that are close enough to fill
    fill_coords = np.column_stack(np.where(~mask & close_to_source))
    
    # Create result array
    result = data.copy()
    
    # If there are cells to fill, perform interpolation
    if len(fill_coords) > 0:
        filled_values = griddata(
            source_coords, 
            source_data, 
            fill_coords, 
            method='nearest',
            fill_value=np.nan
        )
        
        # Fill the result array at the appropriate locations
        fill_mask = ~mask & close_to_source
        result[fill_mask] = filled_values
    
    return result
