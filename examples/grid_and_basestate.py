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
import matplotlib.pyplot as plt
import numpy as np

# Local library
from microhhpy.spatial.vertical_grid import Vertical_grid_2nd
from microhhpy.thermo import Basestate_moist


plt.close('all')

TF = np.float32

ktot = 128
zsize = 3200
dz = zsize / ktot
z = np.arange(dz/2, zsize, dz)

# Define vertical grid, identical to internal definition in MicroHH.
vgrid = Vertical_grid_2nd(z, zsize, remove_ghost=False)

# Define thermodynamic state.
thl = 290 + z * 0.006
qt  = 10e-3 - z * 0.002e-3
pbot = 101300

# Define base state, identical to calculation in MicroHH.
bs = Basestate_moist(thl, qt, pbot, z, zsize, remove_ghost=True, dtype=TF)

# Plot!
plt.figure(layout='tight')

plt.subplot(121)
plt.plot(bs.pref, z)
plt.xlabel('p (Pa)')
plt.ylabel('z (m)')

plt.subplot(122)
plt.plot(bs.rho, z)
plt.xlabel('rho (kg/m3)')
