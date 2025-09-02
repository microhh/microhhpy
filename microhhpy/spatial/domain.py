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
from .projection import Projection
from microhhpy.logger import logger


class Domain:
    def __init__(
            self,
            xsize,
            ysize,
            itot,
            jtot,
            n_ghost=0,
            n_sponge=0,
            lbc_freq=None,
            buffer_freq=None,
            xstart_in_parent=None,
            ystart_in_parent=None,
            parent=None,
            child=None,
            center_in_parent=False,
            lon=None,
            lat=None,
            anchor='center',
            start_date=None,
            end_date=None,
            proj_str=None,
            work_dir=''):
        """
        Domain and nesting specification of single domain.

        NOTE: you can only specify the `lat/lon` location for the outer domain.
              Nests are described relative to their parent using the
              `xstart_in_parent=..` and `ystart_in_parent=..` or `center_in_parent=True` options.

        Parameters:
        ----------
        xsize : float
            Domain size in x-direction (m).
        ysize : float
            Domain size in y-direction (m).
        itot : int
            Number of grid points in x-direction.
        jtot : int
            Number of grid points in y-direction.
        n_ghost : int, optional
            Number of horizontal ghost cells.
        n_sponge : int, optional
            Number of lateral sponge cells.
        lbc_freq : int of float, optional
            Time interval (sec) of lateral boundary updates.
        buffer_freq : int or float, optional
            Timer interval (sec) of buffer updates.
        xstart_in_parent: float, optional, optional
            x-offset in parent domain.
        ystart_in_parent: float, optional, optional
            y-offset in parent domain.
        parent : `Domain` instance, optional, optional
            Parent of current domain.
        child : `Domain` instance, optional, optional
            Child of current domain.
        center_in_parent : bool, optional, optional
            Center current domain in parent domain.
        lon : float, optional, optional
            Longitude of domain (degrees), only for outer domain.
        lat : float, optional, optional
            Latitude of domain (degrees), only for outer domain.
        anchor : str, default: 'center'
            Anchor point of (`lon, lat`), ∈ ('center', 'southwest')
        start_date : datetime instance or None
            Start date of experiment.
        end_date : datetime instance or None
            End date of experiment.
        proj_str : str, optional
            Proj.4 / pyproj projection string for lon/lat <-> x/y transformations.
        work_dir : str, optional
            Work directory/path of experiment.

        Returns:
        -------
        None
        """

        # Checks: child domain should know the relative position to its parent.
        if parent and (xstart_in_parent is None and ystart_in_parent is None and not center_in_parent):
            logger.critical('child domains need to specify the relative location to its parent.')

        self.itot = itot
        self.jtot = jtot

        self.xsize = xsize
        self.ysize = ysize

        self.n_ghost = n_ghost
        self.n_sponge = n_sponge
        self.n_pad = self.n_ghost + 1
        self.n_lbc = self.n_pad + n_sponge

        self.lbc_freq = lbc_freq
        self.buffer_freq = buffer_freq

        self.parent = parent
        self.child = child

        self.proj_str = proj_str
        self.work_dir = work_dir

        self.start_date = start_date
        self.end_date = end_date

        self.dx = xsize / itot
        self.dy = ysize / jtot

        self.dxi = 1/self.dx
        self.dyi = 1/self.dy

        self.x = np.arange(self.dx/2, self.xsize, self.dx)
        self.y = np.arange(self.dy/2, self.ysize, self.dy)

        self.xh = np.arange(0, self.xsize, self.dx)
        self.yh = np.arange(0, self.ysize, self.dy)

        if parent is None:
            self.xstart_in_parent = 0
            self.ystart_in_parent = 0

            self.xoffset = 0
            self.yoffset = 0

        elif center_in_parent:
            self.xstart_in_parent = (parent.xsize - self.xsize) / 2.
            self.ystart_in_parent = (parent.ysize - self.ysize) / 2.

            self.xoffset = parent.xoffset + self.xstart_in_parent
            self.yoffset = parent.yoffset + self.ystart_in_parent

        else:
            self.xstart_in_parent = xstart_in_parent
            self.ystart_in_parent = ystart_in_parent

            self.xoffset = parent.xoffset + self.xstart_in_parent
            self.yoffset = parent.yoffset + self.ystart_in_parent


        # Check: half levels x/y parent and child need to coincide at lateral boundaries child.
        if self.parent is not None:
            if not np.isclose(self.xstart_in_parent % parent.dx, 0):
                logger.critical('invalid starting position x-direction child in parent domain.')
            elif not np.isclose(self.ystart_in_parent % parent.dy, 0):
                logger.critical('invalid starting position y-direction child in parent domain.')
            elif not np.isclose(self.xsize % parent.dx, 0):
                logger.critical('invalid child size in x-direction, not a multiple of parent dx.')
            elif not np.isclose(self.ysize % parent.dy, 0):
                logger.critical('invalid child size in y-direction, not a multiple of parent dy.')


        # Calculate lat/lon of each grid point using provided Proj.4 / PyProj string.
        self.proj = None

        # Outer domain.
        if parent is None and lon is not None and lat is not None:
            if proj_str is None:
                logger.critical('for spatial projections, specify the Proj.4 `proj_str`.')

            self.proj = Projection(
                self.xsize,
                self.ysize,
                self.itot,
                self.jtot,
                lon,
                lat,
                anchor,
                proj_str
            )

        # Inner domain
        elif parent is not None and parent.proj is not None:

            istart_in_parent = int(self.xstart_in_parent / parent.dx)
            jstart_in_parent = int(self.ystart_in_parent / parent.dy)

            lon_sw = self.parent.proj.lon_h[jstart_in_parent, istart_in_parent]
            lat_sw = self.parent.proj.lat_h[jstart_in_parent, istart_in_parent]

            self.proj_str = self.parent.proj_str

            self.proj = Projection(
                    self.xsize,
                    self.ysize,
                    self.itot,
                    self.jtot,
                    lon_sw,
                    lat_sw,
                    'southwest',
                    self.proj_str)


        if self.proj is not None:
            # Create projection with padding for interpolation horizontal ghost cells.
            # Add one extra ghost cell, needed for `u` at east boundary and `v` at north.
            itot_pad = self.itot + 2 * self.n_pad
            jtot_pad = self.jtot + 2 * self.n_pad

            xsize_p = itot_pad * self.dx
            ysize_p = jtot_pad * self.dy

            self.proj_pad = Projection(
                    xsize_p,
                    ysize_p,
                    itot_pad,
                    jtot_pad,
                    self.proj.central_lon,
                    self.proj.central_lat,
                    'center',
                    self.proj_str)

            # Define start/end indices
            self.istart_pad = self.n_pad
            self.iend_pad = self.itot + self.n_pad

            self.jstart_pad = self.n_pad
            self.jend_pad = self.jtot + self.n_pad


def plot_domains(
        domains,
        use_projection=False,
        scatter_lon=[],
        scatter_lat=[],
        scatter_lonlat=False,
        labels=None):
    """
    Plot position of all domains.

    Parameters:
    ----------
    domains : list(Domain)
        List of `Domain` instances.
    use_projection : bool, optional
        Plot domains on map in lon/lat space instead of x/y.
    scatter_lon : list(float)
        Scatter lat/lon points on map as reference.
    scatter_lat : list(float)
        Scatter lat/lon points on map as reference.
    scatter_lonlat : bool, optional
        Scatter half level lon/lat points, to check match parent/child position.

    Returns:
    -------
    None
    """

    # NOTE: not nice to have imports at function level, but plotting related
    #       imports can be a pain in the *** on some supercomputers.
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if labels is None:
        labels = []
        for n,dom in enumerate(domains):
            labels.append(f'#{n}: {dom.xsize/1000}x{dom.ysize/1000} km @ {dom.dx} m.')


    if use_projection:
        """
        Plot domains on map in lon/lat projection.
        """

        # Find max extent domains.
        lon_min = 1e9
        lon_max = -1e9

        lat_min = 1e9
        lat_max = -1e9

        for i,d in enumerate(domains):
            lon_min = min(lon_min, d.proj.lon.min())
            lon_max = max(lon_max, d.proj.lon.max())

            lat_min = min(lat_min, d.proj.lat.min())
            lat_max = max(lat_max, d.proj.lat.max())

        margin = 0.05 * (lon_max - lon_min)

        extent = [lon_min-margin, lon_max+margin, lat_min-margin, lat_max+margin]

        proj = ccrs.LambertConformal(
            central_longitude=0.5*(lon_min + lon_max),
            central_latitude=0.5*(lat_min + lat_max))

        plt.figure(layout='tight')
        ax = plt.subplot(projection=proj)

        for i,d in enumerate(domains):

            ax.plot(d.proj.bbox_lon, d.proj.bbox_lat, label=labels[i], transform=ccrs.PlateCarree())

            if scatter_lonlat:
                plt.scatter(d.proj.lon_h, d.proj.lat_h, s=1, transform=ccrs.PlateCarree())

        for lon, lat in zip(scatter_lon, scatter_lat):
            plt.scatter(lon, lat, transform=ccrs.PlateCarree())

        plt.legend(loc='upper left')

        ax.set_extent(extent)

        # Add coast lines, countries, etc.
        ax.coastlines(resolution='10m', linewidth=0.8, color='0.5')

        countries = cfeature.NaturalEarthFeature(
                category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none', zorder=100)
        ax.add_feature(countries, edgecolor='0.5', linewidth=0.8)

        lakes = cfeature.NaturalEarthFeature(
                category='physical', name='lakes', scale='10m', facecolor='none', zorder=100)
        ax.add_feature(lakes, edgecolor='0.5', linewidth=0.8)

    else:
        """
        Simple plot domains in x/y space.
        """

        plt.figure(layout='tight')
        plt.subplot(aspect='equal')

        for i,d in enumerate(domains):

            bbox_x = [d.xoffset, d.xoffset+d.xsize, d.xoffset+d.xsize, d.xoffset, d.xoffset]
            bbox_y = [d.yoffset, d.yoffset, d.yoffset+d.ysize, d.yoffset+d.ysize, d.yoffset]

            plt.plot(bbox_x, bbox_y, label=f'#{i}')

        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()
