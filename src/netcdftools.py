import netCDF4 as nc
import netcdftime
import pandas as pd
import numpy as np
from scipy import spatial


class MyNetCDF:
    """
    Class for handling general netcdf data
    Currently all the methods are used for time-series single-point data
    TODO – change classes in order to distinguish single-point and multi-point approach
    TODO - to make a constructor class and to inherit other classes from it
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

    def extract_point(self, x, y, time, variables, z=0):
        """
        Extracting a single point from dataset
        """
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
        """
        Converts timestep to datetime object
        :param time_step: an int time step to convert
        :return: formatted as '%Y%m%d' string
        """
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
        Method to get timeseries of time-distributed 3D datasets
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
    """
    TODO – change classes in order to distinguish single-point and multi-point approach
    TODO – handle a multi-variable data
    Child class to handle spatially-distributed multi-point data
    """

    def __init__(self, path, variable=None):
        """
        :param path: path to a .nc file
        :param variable: a data variable (for example ice thickness)

        """
        super().__init__(path)
        self.vector = None

        self.variable = variable
        self.data = None

        if variable:
            self.set_variable()

    def set_variable(self):
        """
        If a variable name provided, initialize data and vector variables
        """
        self.data = self.dset.variables[self.variable][:]  # extract data from nc dataset

        self.mask()
        self.data = np.ma.filled(self.data, np.nan)  # fill data with nan based on mask
        self.vector = self.data.ravel()  # convert data to 1d and store in in vector attribute

    def mask(self):
        """
        TODO - make it more universal
        Generates a mask, based on 'status flag ' vatiable
        :return: masked self.data attribute
        """
        # OSISAF case
        if 'status_flag' in self.dset.variables.keys():
            mask = self.dset.variables['status_flag'][:]
            self.data = np.ma.masked_where(np.logical_or(mask == 100, mask == 101), self.data)
        else:
            pass

    def timelist(self, tname='time'):
        """
        Generates array of datetime.datetime objects, based on dataset time variable
        :param tname: name of a time variable
        :return: array of datetime.datetime objects
        """

        t_unit = self.dset.variables[tname].units  # get unit

        try:
            t_cal = self.dset.variables[tname].calendar
        except AttributeError:  # Attribute doesn't exist
            t_cal = u"gregorian"  # or standard

        try:
            return netcdftime.num2date(self.dset.variables[tname][:],
                                       units=t_unit,
                                       calendar=t_cal)
        except ValueError:
            # TODO - fix it
            raise Exception('no ''since'' in unit_string')

    def to_df(self):
        """
        Generates a dataframe, using existing class attributes
        :return: pandas dataframe with columns time, x, y, value
        """
        time = self.timelist()
        time = np.repeat(time, (np.shape(self.data))[1] * np.shape(self.data)[2])
        df = pd.DataFrame({'time': time,
                           'x': np.indices(np.shape(self.data))[1].ravel(),
                           'y': np.indices(np.shape(self.data))[2].ravel(),
                           self.variable: self.vector})
        return df
