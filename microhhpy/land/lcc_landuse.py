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
import rioxarray as rxr

# Local library
from microhhpy.logger import logger


def read_lcc(geotiff_file, lon_0, lon_1, lat_0, lat_1):
    """
    Read LCC GeoTIFF file, and sub-select
    lat/lon bounding box to decrease memory usage.
    """

    # Read full (global) dataset, and slice requested area.
    da_full = rxr.open_rasterio(geotiff_file, engine='rasterio')
    da_sub = da_full.sel(x=slice(lon_0, lon_1), y=slice(lat_1, lat_0))
    da_full.close()

    # Rename dimensions to stay in line with CORINE dataset.
    da_sub = da_sub.rename( {'x': 'lon', 'y': 'lat'} )

    # Remove empty dimensions:
    da_sub = da_sub.squeeze()

    return da_sub


def plot_lcc(da):
    """
    Plot LCC land-use using "official" colormap
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Read table with vegetation names, codes, and RGB colors
    code,r,g,b = np.loadtxt('lcc_categories.txt', delimiter=',', usecols=[0,2,3,4], unpack=True)
    name = np.loadtxt('lcc_categories.txt', delimiter=',', usecols=1, dtype=str)

    colors = np.zeros((r.size, 4))
    colors[:,0] = r/255
    colors[:,1] = g/255
    colors[:,2] = b/255
    colors[:,3] = 1

    # Generate colormap.
    cmap = LinearSegmentedColormap.from_list('corine', colors, N=colors.shape[0])

    # Translate codes into continuous range of numbers for plotting.
    lu = da.values[:,:].copy()
    for i,c in enumerate(code):
        lu[lu==c] = i

    # Plot!
    plt.figure(figsize=(8,5))
    plt.pcolormesh(da.lon, da.lat, lu, cmap=cmap, vmin=-0.5, vmax=r.size-0.5)
    cbar=plt.colorbar()
    cbar.set_ticks(np.arange(0, 23))
    cbar.set_ticklabels(name)
    cbar.ax.tick_params(labelsize=8)
    plt.xlabel('longitude (deg)')
    plt.ylabel('latitude (deg)')
    plt.tight_layout()


def get_lcc_ifs_lut():
    """
    Generate lookup table for LCC to IFS land-use conversion.
    """

    lut = dict()
    lut[0]   = -9999 # No data

    # Closed forest
    lut[111] = 2     # evergreen needle -> evergreen needle
    lut[113] = 3     # deciduous needle -> deciduous needle
    lut[112] = 5     # evergreen broad lead -> evergreen broad leaf
    lut[114] = 4     # deciduous broad leaf -> deciduous broad leaf
    lut[115] = 17    # mixed -> mixed forest wood
    lut[116] = 17    # unknown -> mixed forest wood

    # Open forest. Not sure what to do with these types, according to the
    # documentation these are 15-70% trees with grass/shrubs in between...
    lut[121] = 18    # evergreen needle -> interrupted forest
    lut[123] = 18    # deciduous needle -> interrupted forest
    lut[122] = 18    # evergreen broad leaf -> interrupted forest
    lut[124] = 18    # deciduous broad leaf -> interrupted forest
    lut[125] = 18    # mixed -> interrupted forest
    lut[126] = 18    # open forest -> interrupted forest

    # Low vegetation / misc
    lut[20]  = 16    # shrubs -> deciduous shrubs (could also be 15 = evergreen)
    lut[30]  = 1     # herbaceous vegetation -> short grass
    lut[90]  = 1     # herbaceous wetland -> short grass (????)
    lut[100] = 1     # moss and lichen -> short grass (???!??!!!)
    lut[60]  = 7     # bare / sparse veg -> desert
    lut[40]  = 0     # cultivated cropland -> crops / mixed farming
    lut[50]  = 20    # urban -> urban
    lut[70]  = 11    # snow and ice -> ice caps / glaciers
    lut[80]  = 13    # water bodies -> inland water
    lut[200] = 14    # open sea -> ocean

    return lut


def lcc_to_ifs_ids(id_lcc):
    """
    Transform LCC vegetation types to IFS types.
    """

    id_ifs = np.zeros_like(id_lcc).astype(np.int32)
    id_ifs[:] = -9999

    lut = get_lcc_ifs_lut()

    for id in np.unique(id_lcc):
        id_ifs[id_lcc == id] = lut[id]

    return id_ifs