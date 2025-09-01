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

# Molecular masses chemical species.
# From: http://dx.doi.org/10.5194/gmd-8-975-2015-supplement
xm_cams = {
    'o3': 48.0,
    'no2': 46.0,
    'co': 28.0,
    'hcho': 30.0,
    'h2o2': 34.0,
    'hno3': 63.0,
    'ch3ooh': 48.0,
    'par': 12.0,
    'c2h4': 28.0,
    'ole': 24.0,
    'rooh': 47.0,
    'c5h8': 68.1,
    'no': 30.0,
    'ho2': 33.0,
    'ch3o2': 47.0,
    'oh': 17.0,
    'no3': 62.0,
    'n2o5': 76.0,
    'ch3oh': 31.01,
    'c2h6': 30.02,
    'c2h5oh': 46.02,
    'c3h8': 44.03,
    'c3h6': 42.03,
    'c10h16': 136.0,
    'c2o3': 75.0,
    'aco2': 58.0,
    'ic3h7o2': 75.0,
    'hypropo2': 91.0,
    # Bonus:
    'air': 28.9647,
    'h2o': 18.01528,
    'co2': 44.0095
}
