import os

import netCDF4 as nc
import numpy as np

import cfg
#from src.feature_table import FeatureTable


def load(variable, year):
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    path = '{}/{}_{}.npy'.format(abspath, variable, year)

    try:
        data = np.load(path)
        return data
    except FileNotFoundError:

        var = next((item for item in cfg.var_dict if item['variable'] == variable), None)
        if var is None:
            raise Exception('Variable is not found in config file')

        path_nc = '{}{}/{}'.format(var['path'], year, var['file_mask'].format(year))

        ds = nc.Dataset(path_nc)
        data = ds[variable][:]

        # OSISAF case
        if 'status_flag' in ds.variables.keys():
            mask = ds.variables['status_flag'][:]
            data = np.ma.masked_where(np.logical_or(mask == 100, mask == 101), data)
        else:
            pass

        try:
            data = np.ma.filled(data, np.nan)
        except:
            pass

        np.save(path, data)
        return data


def load_features(y_var, X_vars, years):
    X_arr = []
    for var in X_vars:

        to_append = load_variable_years(var, years)
        X_arr.append(to_append)

    y_arr = load_variable_years(y_var, years)
    return y_arr, X_arr



def load_variable_years(variable, years, point=None):
    data = []
    for year in years:
        d = load(year=year, variable=variable)
        if point:
            data.append(d[:, point[0], point[1]])
        else:
            data.append(d)
    out_data = np.concatenate([*data])
    return out_data