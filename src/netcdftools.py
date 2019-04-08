import netCDF4 as nc
import netcdftime
import pandas as pd
import numpy as np
from scipy import spatial

import warnings


class MyNetCDF:
    """
    TODO: implement xarray
    Some instruments to work with NetCDF files
    """

    def __init__(self, path):

        self.dset = self.open_netcdf(path)
        self.data = None

    def open_netcdf(self, path):
        return nc.Dataset(path)

    def coords_to_index(self, coords):
        """
        The idea is to make a table, where all the model grid points are stored in a following manner:
        latitide - longitude - x index - y index, аnd make query to this table
        """

        try:
            lat = self.dset.variables['nav_lat'][:]
            lon = self.dset.variables['nav_lon'][:]
        except:
            lat = self.dset.variables['latitude'][:]
            lon = self.dset.variables['longitude'][:]

        # The following code returns 2x1d arrays for the lat-lon mesh
        lat_mesh = np.dstack(np.meshgrid(np.arange(np.shape(lat)[0]),
                                         np.arange(np.shape(lat)[1]),
                                         indexing='ij'))[:, :, 0].ravel()

        lon_mesh = np.dstack(np.meshgrid(np.arange(np.shape(lat)[0]),
                                         np.arange(np.shape(lat)[1]),
                                         indexing='ij'))[:, :, 1].ravel()
        # stack all the 1d arrays to the table
        array = np.column_stack((lat.ravel(),
                                 lon.ravel(),
                                 lat_mesh,
                                 lon_mesh))

        latlonarr = array[:, [0, 1]]
        '''
         Here the KD-tree algorythm is used for finding the closest spatial point (nearest neighbour)
         More information about the algorithm
         https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
         input: array(2d array of points to search), query (list-like coordinates to find neighbour)
        '''
        tree = spatial.KDTree(latlonarr, leafsize=100)
        result = tree.query(coords)[1]

        idxs = array[result][[2, 3]].astype(int)
        true_coords = array[result][[0, 1]]

        return idxs, true_coords


class NemoNC(MyNetCDF):

    def extract_point(self, x, y, time, variables, z=0):

        out_list = []
        for var in variables:
            data = self.dset.variables[var]

            if len(data.dimensions) == 3:
                value = data[time, x, y]
            else:
                value = data[time, z, x, y]

        out_list.append(value)
        return out_list

    def file_time(self, time_step):

        try:
            tname = 'time'
            nctime = self.dset.variables[tname][time_step]  # get values
            t_unit = self.dset.variables[tname].units  # get unit
        except KeyError:
            tname = 'time'
            nctime = self.dset.variables[tname][time_step]  # get values
            t_unit = self.dset.variables[tname].units  # get unit

        try:
            t_cal = self.dset.variables[tname].calendar
        except AttributeError:  # Attribute doesn't exist
            t_cal = u"gregorian"  # or standard

        datevar = netcdftime.num2date(nctime, units=t_unit, calendar=t_cal)

        datestr = datevar.strftime('%Y%m%d')
        return datestr

    def get_timeseries(self, lat, lon, variables):
        """
        Method to get timeseries of time-distributed 4D datasets
        :param lat: integer, latitude
        :param lon: integer, longitude
        :param variables: list of variables to extract
        :return: dataframe with time-distributed data
        """
        idxs, truecoords = self.coords_to_index([lat, lon])
        data = []
        time = []
        for i, date in enumerate(self.dset.variables['time'][:]):
            data.append(self.extract_point(x=idxs[0], y=idxs[1], time=i, variables=variables))
            time.append(self.file_time(i))

        df = pd.DataFrame(data=data, columns=variables)
        df['Date'] = time
        df.set_index(pd.to_datetime(df['Date'], format='%Y-%m-%d'), inplace=True)
        df.drop('Date', inplace=True, axis=1)

        return df


class SatelliteNC(MyNetCDF):

    def __init__(self, path, variable=None):
        super().__init__(path)
        self.vector = None

        self.variable = variable
        if variable:
            self.data = self.get_variable(variable)

            self.mask()
            self.data = np.ma.filled(self.data, np.nan)
            self.to_1d()

        self.dframe = None

    def get_variable(self, varname):
        self.data = self.dset.variables[varname][:]
        return self.data

    def to_1d(self):
        self.vector = self.data.ravel()

    def mask(self):
        try:
            mask = self.dset.variables['status_flag'][:]
            self.data = np.ma.masked_where(np.logical_or(mask == 100, mask == 101), self.data)
        except KeyError:
            warnings.warn('Mask was not set')

    def to_df(self):

        tstep = np.indices(np.shape(self.data))[0].ravel()
        time = map(self.file_time, tstep)
        time = np.fromiter(time, dtype=np.int)
        df = pd.DataFrame({'time': time,
                           'x': np.indices(np.shape(self.data))[1].ravel(),
                           'y': np.indices(np.shape(self.data))[2].ravel(),
                           self.variable: self.vector})
        return df


class Osisaf(MyNetCDF, SatelliteNC):
    pass

