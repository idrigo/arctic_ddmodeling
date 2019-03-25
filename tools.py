import netCDF4 as nc
import netcdftime
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from scipy import spatial


class MyNetCDF:
    """
    TODO: implement xarray
    Some instruments to work with NetCDF files
    """
    def __init__(self, dset):
        self.dset = dset

        self.x = dset.dimensions['x'].size
        self.y = dset.dimensions['y'].size
        try:
            self.z = dset.dimensions['deptht'].size
        except KeyError:
            pass

    def coords_to_index(self, coords):
        """
        The idea is to make a table, where all the model grid points are stored in a following manner:
        latitide - longitude - x index - y index And make query to this table
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


class Preprocessing:

    def __init__(self, df=None):
            self.df = df

    def load_csv(self, filepath, continuous_check = True):
        ds = pd.read_csv(filepath, sep='\t')
        ds.set_index(pd.to_datetime(ds['Date'], format='%Y-%m-%d'), inplace=True)
        ds.drop('Date', inplace=True, axis=1)

        self.df = ds
        if continuous_check:
            self.continuous_check()

        return self.df

    def load_pickle(self, filepath, continuous_check=False):
        ds = pd.read_pickle(filepath)
        self.df = ds
        if continuous_check:
            self.continuous_check()
        return ds

    def continuous_check(self, method='ffill'):
        """
        Method to check the data for missing dates
        :param method: {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}
        :return:
        """
        idx = pd.date_range(start=self.df.index[0],
                            end=self.df.index[0] + pd.offsets.YearEnd(),
                            freq='D')
        ds = self.df.reindex(idx, fill_value=np.nan)
        print('Found {} gaps'.format(ds.isna().sum().values[0]))

        ds.fillna(method=method, inplace=True)

        self.df = ds
        return

    def velocity_module(self, x):  # simple function to convert UV velocity to velocity module
        return np.sqrt(x[0] ** 2 + x[1] ** 2)
