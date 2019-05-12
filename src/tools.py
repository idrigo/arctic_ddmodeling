import numpy as np
import pandas as pd

class Preprocessing:
    """
    Class to handle csv files, generated from netcdf
    """
    def __init__(self, df=None):
        self.df = df

    def load_csv(self, filepath, continuous_check=True):
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


