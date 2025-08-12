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
from rasterio.warp import transform

# Local library
from microhhpy.logger import logger


def read_corine(geotiff_file, lon_0, lon_1, lat_0, lat_1):
    """
    Read CORINE GeoTIFF file, and sub-select
    lat/lon bounding box to decrease memory usage.
    """

    def calc_latlon(da):
        ny, nx = len(da['y']), len(da['x'])
        x, y = np.meshgrid(da['x'], da['y'])

        crs = da.rio.crs.to_proj4()
        lon, lat = transform(crs, {'init': 'EPSG:4326'}, x.flatten(), y.flatten())

        lon = np.asarray(lon).reshape((ny, nx))
        lat = np.asarray(lat).reshape((ny, nx))

        # Add to dataset:
        da.coords['lon'] = (('y', 'x'), lon)
        da.coords['lat'] = (('y', 'x'), lat)

    lon_c = 0.5*(lon_0 + lon_1)
    lat_c = 0.5*(lat_0 + lat_1)

    # Read GEOTIFF data with xarray:
    da_full = rxr.open_rasterio(geotiff_file)

    # Generate coordinates for sub-sampled field
    da_full_ss = da_full[:,::20,::20]
    calc_latlon(da_full_ss)

    # Get x/y bounds for sub-selection full dataset:
    ji = np.sqrt((da_full_ss.lon-lon_c)**2 + (da_full_ss.lat-lat_c)**2).argmin()
    j,i = np.unravel_index(ji, da_full_ss.lon.shape)

    i0 = int(np.abs(da_full_ss.lon[j,:]-lon_0).argmin())
    i1 = int(np.abs(da_full_ss.lon[j,:]-lon_1).argmin())

    j0 = int(np.abs(da_full_ss.lat[:,i]-lat_0).argmin())
    j1 = int(np.abs(da_full_ss.lat[:,i]-lat_1).argmin())

    x0 = int(da_full_ss.x[i0])
    x1 = int(da_full_ss.x[i1])

    y0 = int(da_full_ss.y[j0])
    y1 = int(da_full_ss.y[j1])

    # Sub-selections area. Full dataset = 46000 x 65000 grid points!:
    da_sub = da_full.sel(x=slice(x0, x1), y=slice(y1, y0))

    # Add lat/lon coordinates
    calc_latlon(da_sub)

    # Remve empty dimensions
    da_sub = da_sub.squeeze()
    da_full = da_full.squeeze()

    return da_sub, da_full


def plot_corine(da, ss=1):
    """
    Plot CORINE dataset, possibly sub-sampled at every `ss`th grid point.
    """
    import matplotlib.pyplot as pl
    from matplotlib.colors import LinearSegmentedColormap

    # Sub-sample data, to prevent memory -> boom.
    da_lr = da[::ss, ::ss]

    # Get bounds for plotting with `imshow`,
    # which is a lot faster than `pcolormesh`:
    x0 = int(da_lr.x[0])/1000
    x1 = int(da_lr.x[-1])/1000
    y0 = int(da_lr.y[-1])/1000
    y1 = int(da_lr.y[0])/1000

    cols_urban  = pl.cm.Greys_r(np.linspace(0, 0.7, 11))
    cols_nature = pl.cm.Greens(np.linspace(0.3, 1, 23))
    cols_water  = pl.cm.Blues(np.linspace(0.3, 1, 10))
    colors = cols_urban
    colors = np.append(colors, cols_nature, axis=0)
    colors = np.append(colors, cols_water, axis=0)
    colors[26,:] = np.array([66,0,87,255])/255.

    cmap = LinearSegmentedColormap.from_list('corine', colors, N=colors.shape[0])

    pl.figure()
    pl.imshow(da_lr, extent=[x0, x1, y0, y1], vmin=0.5, vmax=44.5, cmap=cmap, interpolation='nearest')
    pl.xlabel('x (km)')
    pl.ylabel('y (km)')
    pl.colorbar()


def get_corine_ifs_lut():
    """
    Lookup table for CORINE to IFS land-use conversion.
    """
    lut = np.array([
        21,   # 1.  Continuous urban fabric -> urban
        21,   # 2.  Discontinuos urban fabric -> urban
        21,   # 3.  Industrial of commercial units -> urban
        21,   # 4.  Road and rail networks and associated land -> urban
        21,   # 5.  Port areas -> urban
        21,   # 6.  Airports -> urban
        8,    # 7.  Mineral extraction sies -> desert
        8,    # 8.  Dump sites -> desert ???
        21,   # 9.  Construction sites -> urban
        17,   # 10. Green urban areas -> deciduous shrubs ???
        2,    # 11. Sport and leisure facilities -> short grass
        1,    # 12. Non-irrigated arable land -> crops, mixed-farming
        10,   # 13. Permanently irrigated land -> irrigated crops
        10,   # 14. Rice fields -> irrigated crops
        17,   # 15. Vineyards -> deciduous shrubs
        17,   # 16. Fruit trees and berry plantations -> deciduous shrubs
        17,   # 17. Olive groves -> deciduous shrubs
        2,    # 18. Pastures -> short grass
        1,    # 19. Annual crops associated with permanent crops -> crops, mixed-farming
        1,    # 20. Complex cultivation patterns -> crops, mixed-farming
        1,    # 21. Land principally occupied by agriculture with.... -> crops, mixed-farming
        19,   # 22. Agro-forestry areas -> interrupted forest
        5,    # 23. Broad-leaved forest -> deciduous broadleaf
        3,    # 24. Coniferous forest -> evergreen needleleaf
        18,   # 25. Mixed forest -> mixed forest-wood
        7,    # 26. Natural grasslands -> tall grass
        9,    # 27. Moors and heathland -> tundra
        6,    # 28. Sclerophyllous vegetation -> evergreen broadleaf
        19,   # 29. Transitional woodland-shrub -> interrupted forest
        8,    # 30. Beaches dunes sands -> desert
        8,    # 31. Bare rocks -> desert
        9,    # 32. Sparsely vegetated areas -> tundra
        9,    # 33. Burnt areas -> tundra
        12,   # 34. Glaciers and perpetual snow -> ice caps and glaciers
        13,   # 35. Inland marshes -> bogs and marshes
        13,   # 36. Peat bogs -> bogs and marshes
        13,   # 37. Salt marshes -> bogs and marshes
        14,   # 38. Saline -> inland water
        14,   # 39. Intertidal flats -> inland water
        14,   # 40. Water courses -> inland water
        14,   # 41. Water bodies -> inland water
        20,   # 42. Coastal lagoons -> water-land mixtures
        20,   # 43. Estuaries -> water-land mixtures
        15,   # 44. Sea and ocean -> ocean
        -9999]) # 45. NODATA

    # FORTRAN to C/Python indexing:
    lut[:] -= 1

    return lut


#def print_corine_ifs_lut():
#    """
#    Print the Corine to IFS vegetation lookup table
#    """
#    # Read name of land-use, index in dataset, and code:
#    names = np.loadtxt('corine_categories.txt', dtype='str', usecols=[6], delimiter=',', unpack=True)
#    index, code = np.loadtxt('corine_categories.txt', dtype='int', usecols=[0,1], delimiter=',', unpack=True)
#
#    # Lookup table with IFS vegetation:
#    lut = get_corine_ifs_lut()
#
#    print('Corine ---> IFS')
#    print('---------------')
#    for i in range(lut.size):
#        i_ifs = lut[i]-1
#        if i_ifs >= 0:
#            print(i+1, names[i], '--->', ifs_vegetation['name'][i_ifs])
#        else:
#            print(i+1, names[i], '---> NO DATA')


def corine_to_ifs_ids(id_corine):
    """
    Transform Corine vegetation types to IFS types.
    """

    id_ifs = np.zeros_like(id_corine, dtype=np.int32)
    id_ifs[:] = -9999

    lut = get_corine_ifs_lut()

    for id in np.unique(id_corine):
        if id == -128:
            id_ifs[id_corine == id] = 15    # Ocean/no-data
        else:
            id_ifs[id_corine == id] = lut[id-1]

    return id_ifs