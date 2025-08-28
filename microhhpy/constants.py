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

kappa = 0.4           # von Karman constant (-)
grav = 9.81           # Gravitational acceleration (m s-2)
e_rot = 7.2921e-5     # Earth rotation rate (s-1)
Rd = 287.04           # Gas constant for dry air (J K-1 kg-1)
Rv = 461.5            # Gas constant for water vapor (J K-1 kg-1)
cp = 1005             # Specific heat of air at constant pressure (J kg-1 K-1)
Lv = 2.501e6          # Latent heat of vaporization (J kg-1)
Lf = 3.337e5          # Latent heat of fusion (J kg-1)
Ls = Lv + Lf          # Latent heat of sublimation (J kg-1)
T0 = 273.15           # Freezing / melting temperature (K)
p0 = 1.e5             # Reference pressure (Pa)
rho_w = 1.e3          # Density of water (kg m-3)
rho_i = 7.e2          # Density of ice   (kg m-3)
mu0_min = 1e-6        # Minimum value used for cos(sza)
sigma_b = 5.67e-8     # Boltzmann constant (W m-1 K-1)
ep = Rd / Rv          # Ratio gas constants dry air and water vapor (-)

mm_air = 28.9647      # Molar mass of dry air (kg kmol-1)
mm_h2o = 18.01528     # Molar mass of H2O (kg kmol-1)
mm_co2 = 44.0095      # Molar mass of CO2 (kg kmol-1)
mm_no  = 30.01        # Molar mass of NO (kg kmol-1)
mm_no2 = 46.0055      # Molar mass of NO2 (kg kmol-1)
mm_co  = 28.01        # Molar mass of CO (kg kmol-1)