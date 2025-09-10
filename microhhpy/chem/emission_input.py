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
from numba import njit
from scipy.optimize import curve_fit

# Local library


@njit
def _pow2(x):
    return x*x


@njit
def _add_source_kernel(
        field,
        t,
        strength,
        x0,
        y0,
        z0,
        sigma_x,
        sigma_y,
        sigma_z,
        x,
        y,
        z,
        dx,
        dy,
        dz,
        rho_ref):

    """
    Fast Numba kernel to add source as Gaussian blob to field,
    and normalize it to get a net `strength` as sum over blob.
    """

    nt, ktot, jtot, itot = field.shape

    raw_sum = 0.
    for k in range(ktot):

        # Limit vertical extent to +/- 4*sigma_z. This allows us
        # to clip the input fields, reducing their size.
        if (z[k] > z0-4*sigma_z) and (z[k] < z0+4*sigma_z):

            for j in range(jtot):
                for i in range(itot):

                    blob_norm = np.exp(
                            - _pow2(x[i]-x0)/_pow2(sigma_x)
                            - _pow2(y[j]-y0)/_pow2(sigma_y)
                            - _pow2(z[k]-z0)/_pow2(sigma_z))

                    raw_sum += blob_norm * dx * dy * dz[k] * rho_ref[k]

    scaling = strength / raw_sum

    for k in range(ktot):

        # Limit vertical extent to +/- 4*sigma_z. This allows us
        # to clip the input fields, reducing their size.
        if (z[k] > z0-4*sigma_z) and (z[k] < z0+4*sigma_z):

            for j in range(jtot):
                for i in range(itot):

                    field[t,k,j,i] += scaling * np.exp(
                            - _pow2(x[i]-x0)/_pow2(sigma_x)
                            - _pow2(y[j]-y0)/_pow2(sigma_y)
                            - _pow2(z[k]-z0)/_pow2(sigma_z))


class Emission_input:
    def __init__(
            self,
            fields,
            times,
            x,
            y,
            z,
            dz,
            rho_ref,
            dtype=np.float64):
        """
        Help class to define point source emissions.
        Emissions can be added to a single grid point, or as Gaussian "blobs".

        After defining an `emiss = Emission_input(...)` object, emissions can be added as:
            `emiss.add_gaussian(field='s1', strength=1, time=0, x0=400, y0=800, z0=50, sigma_x=50, sigma_y=50, sigma_z=25)`
        or:
            `emiss.add_point(field='s1', strength=1, time=0, x0=400, y0=800, z0=50)`

        Parameters:
        ----------
        fields : list(str)
            List with scalar fields that have an emission.
        times : np.ndarray, shape (1,)
            Array with output times (s).
        x : np.ndarray, shape (1,)
            Array with x-coordinates grid (m).
        y : np.ndarray, shape (1,)
            Array with y-coordinates grid (m).
        z : np.ndarray (shape 1,)
            Array with z-coordinates grid (m).
        dz : np.ndarray shape (1,)
            Array with full level vertical grid spacing (m).
        rho_ref : np.ndarray shape (1,)
            Array with base state density (kg m-3).
        dtype : np.float32 or np.float64
            Datatype used by MicroHH.
        """

        if isinstance(times, list):
            times = np.array(times)

        self.times = times
        self.fields = fields
        self.rho_ref = rho_ref

        self.nt = times.size

        self.x = x
        self.y = y
        self.z = z

        self.itot = x.size
        self.jtot = y.size
        self.ktot = z.size

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = dz

        # Create 3D emission fields.
        self.data = {}
        for field in self.fields:
            self.data[field] = np.zeros((self.nt, self.ktot, self.jtot, self.itot), dtype=dtype)

        self.is_clipped = False


    def get_index(self, time):
        """
        Get index of `time` in global time array.
        """
        t = np.argmin(np.abs(self.times - time))

        if np.abs(self.times[t] - time) > 1e-6:
            raise Exception(f'Can not find time {time} in {self.times}.')

        return t


    def add_gaussian(self, field, strength, time, x0, y0, z0, sigma_x, sigma_y, sigma_z):
        """
        Add single point source emission, spread out over a Gaussian blob defined by `sigma_xyz`.

        Parameters
        ----------
        field : string
            Emission field name.
        strength : float
            Emission strength (kg s-1).
        time : int
            Emission time (s).
        x0 : float
            Center (x) of Gaussian blob (m).
        y0 : float
            Center (y) of Gaussian blob (m).
        z0 : float
            Center (z) of Gaussian blob (m) .
        sigma_x : float
            Std.dev of Gaussian blob in x-direction (m).
        sigma_y : float
            Std.dev of Gaussian blob in y-direction (m).
        sigma_z : float
            Std.dev of Gaussian blob in z-direction (m).
        """

        t = self.get_index(time)

        _add_source_kernel(
                self.data[field],
                t,
                strength,
                x0,
                y0,
                z0,
                sigma_x,
                sigma_y,
                sigma_z,
                self.x,
                self.y,
                self.z,
                self.dx,
                self.dy,
                self.dz,
                self.rho_ref)


    def add_point(self, field, strength, time, x0, y0, z0):
        """
        Add single point source emission, to single grid point.

        Parameters
        ----------
        field : string
            Emission field name.
        strength : float
            Emission strength (kg s-1).
        time : int
            Emission time (s).
        x0 : float
            Center (x) of emission (m).
        y0 : float
            Center (y) of emission (m).
        z0 : float
            Center (z) of emission (m) .
        """

        t = self.get_index(time)

        i = np.abs(self.x - x0).argmin()
        j = np.abs(self.y - y0).argmin()
        k = np.abs(self.z - z0).argmin()

        volume = self.dx * self.dy * self.dz[k]
        self.data[field][t,k,j,i] += strength / (volume * self.rho_ref[k])


    def add_manual(self, field, value, time, x0, y0, z0):
        """
        Add single point source, without normalisation.

        Parameters
        ----------
        field : string
            Emission field name.
        value : float
            Emission value (..).
        time : int
            Emission time (s).
        x0 : float
            Center (x) of emission (m).
        y0 : float
            Center (y) of emission (m).
        z0 : float
            Center (z) of emission (m) .
        """

        t = self.get_index(time)

        i = np.abs(self.x - x0).argmin()
        j = np.abs(self.y - y0).argmin()
        k = np.abs(self.z - z0).argmin()

        self.data[field][t,k,j,i] += value


    def clip(self):
        """
        Clip 3D fields to required vertical extent.
        This automatically determines the highest emission height.
        """
        self.kmax = 0

        # Find max height over all fields.
        for field, emission in self.data.items():
            # Take sum over time and x,y dimensions to get vertical profile.
            emiss_sum = emission[:,:,:,:].sum(axis=(0,2,3))

            # Max height where total emission is positive.
            if emiss_sum.max() > 0:
                emiss_pos = np.where(emiss_sum > 0)[0]
                self.kmax = max(self.kmax, emiss_pos[-1]+1)

        # Clip fields.
        for field, emission in self.data.items():
            self.data[field] = emission[:,:self.kmax,:,:]

        print(f'Max emission height = {self.z[self.kmax]} m., ktot={self.kmax}')

        self.is_clipped = True


    def to_binary(self, path):
        """
        Save all fields in binary format for MicroHH.
        """

        if not self.is_clipped:
            print('WARNING: saving unclipped fields!')

        for name, fld in self.data.items():
            for t,time in enumerate(self.times):
                fld[t,:].tofile(f'{path}/{name}_emission.{time:07d}')


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0)**2 / sigma**2)

def fit_gaussian_curve(x, y):
    """
    Fit Gaussian curve through profile.

    `y = H + A * exp(-(x-x0)**2 / sigma**2)`

    Parameters:
    ----------
    x : np.ndarray(float, ndim=1)
        x-coordinates.
    y : np.ndarray(float, ndim=1)
        y-coordinates.

    Returns:
    -------
    params : dict
        Dictionary with curve fitted parameters.
    """
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return dict(H=popt[0], A=popt[1], x0=popt[2], sigma=popt[3])
