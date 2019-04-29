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


def load_features(X_vars, y_var, years, point, feature_table, autoreg=False):
    """
    :param X_vars: list of feature variables
    :param y_var: target variable
    :param years: range of years
    :param point: tuple or list of (x,y) point coordinates
    :param feature_table: an instance of a FeatureTable class with given dx,dy,dt parameters
    :return:
    """
    X = []
    n = 0
    if y_var not in X_vars:
        X_vars.append(y_var)
    for var in X_vars:
        print("Loading {}".format(var))
        data = None
        for year in years:
            to_append = load(year=year, variable=var)
            try:
                data = np.append(data, to_append, axis=0)
            except:
                data = to_append

        if var == y_var:
            y_vec = data[:, point[0], point[1]]
            if autoreg:
                # TODO – smth wrong
                Y = feature_table.gen_matrix(data=data, x=point[0], y=point[1], autoreg=False)
        else:
            X_var = feature_table.gen_matrix(data=data, x=point[0], y=point[1])
            X.append(X_var)
            n += 1

    X = np.hstack([*X])

    if autoreg:
        X = np.column_stack([X, Y])


    print("Total features {}, X size {}".format(n, np.shape(X)))
    return y_vec, X
