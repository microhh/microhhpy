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
import pandas as pd
import numpy as np

# Local librariy

def rad_to_deg(rad):
    return 360./(2*np.pi)*rad

def deg_to_rad(deg):
    return 2.*np.pi/360.*deg

def calc_zenith_angle(lat, lon, year, day_of_year, seconds_since_midnight):
    """
    cos(sza) function from IFS

    Based on: Paltridge, G. W. and Platt, C. M. R. (1976).
              Radiative Processes in Meteorology and Climatology.
              Elsevier, New York, 318 pp.

    Parameters:
    ----------
    lat : float
        Latitude (degrees)
    lon : float
        Longitude (degrees)
    year: int
        Year
    day_of_year : int
        Day of the year, 1-based (01-01 = 1)
    seconds_since_midnight : int
        Seconds since midnight.

    Returns:
    -------
    cos_sza : float
        Cosine of solar zenith angle
    """

    # Account for leap year
    if year%4 == 0 and (year%100 != 0 or year%400 == 0):
        days_per_year = 366
    else:
        days_per_year = 365

    # DOY in IFS code is zero based:
    doy = day_of_year-1

    # Lat/lon in radians
    radlat = lat*np.pi/180
    radlon = lon*np.pi/180

    # DOY in range (0,2*pi)
    doy_pi = 2*np.pi*doy/days_per_year

    # Solar declination angle
    declination_angle = \
            0.006918 - 0.399912 * np.cos(doy_pi) + 0.070257 * np.sin(doy_pi) \
            -0.006758 * np.cos(2*doy_pi) + 0.000907 * np.sin(2*doy_pi) \
            -0.002697 * np.cos(3*doy_pi) + 0.00148  * np.sin(3*doy_pi)

    # Hour angle in radians, using true solar time
    a1 = (1.00554 * doy -  6.28306) * np.pi/180
    a2 = (1.93946 * doy + 23.35089) * np.pi/180
    a3 = (7.67825 * np.sin(a1) + 10.09176 * np.sin(a2)) / 60.

    hour_solar_time = (seconds_since_midnight/3600) - a3 + radlon * (180./np.pi/15.0)
    hour_angle = (hour_solar_time-12.0)*15.0*(np.pi/180.)

    # Cosine of solar zenith angle
    cos_zenith = np.sin(radlat)*np.sin(declination_angle) + \
                 np.cos(radlat)*np.cos(declination_angle) * np.cos(hour_angle)

    return cos_zenith


def calc_zenith_angles(lat, lon, dates):
    """
    Calculate cosine of zenith angle for a list or array of dates.
    
    Parameters:
    ----------
    lat : float
        Latitude (degrees)
    lon : float
        Longitude (degrees)
    dates : list
        List or array with dates. Can be `datetime`, `np.datetime64`, or `pd.Timestamp`.

    Return:
    ------
    cos_sza : np.array(float)
        Array with cosine of zenith angle values.
    """
    # Convert to pandas DatetimeIndex.
    if hasattr(dates, 'to_pydatetime'):
        dt_objects = dates.to_pydatetime()
    else:
        dt_objects = pd.to_datetime(dates).to_pydatetime()
    
    cos_sza = np.zeros(len(dt_objects))
    for i, date in enumerate(dt_objects):
        seconds = date.hour * 3600 + date.minute * 60 + date.second
        cos_sza[i] = calc_zenith_angle(lat, lon, date.year, date.timetuple().tm_yday, seconds)

    return cos_sza
