import netCDF4 as nc
import numpy as np
from scipy import interpolate
import calendar
import logging

try:
    from src import cfg
except:
    import cfg


def load(variable, year):
    abspath = cfg.processed_data_path
    path = '{}/{}_{}.npy'.format(abspath, variable, year)

    try:
        data = np.load(path, allow_pickle=True)
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

        # check if there is not enough timesteps in netcdf and add dummy layer if needed
        days = 366 if calendar.isleap(year) else 365
        if data.shape[0] != days:
            a = np.empty(data.shape[1:3])
            a[:] = np.nan
            a = np.moveaxis(np.atleast_3d(a), -1, 0)
            data = np.append(data, a, axis=0)

        np.save(path, data)
        return data


def load_features(y_var, X_vars, years, point=None):
    X_arr = []
    for var in X_vars:
        to_append = load_variable_years(var, years, point)
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


def save(self):  # todo - доделать
    import datetime
    import os
    time_now = datetime.datetime.now().strftime("%m%d_%H%M")
    logging.info('Saving output to file')

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'output'))
    self.out.dump(os.path.join(path, 'res_{}.npy'.format(time_now)))

    self.coeffcients_out.dump((os.path.join(path, 'coeffs_{}.npy'.format(time_now))))
    logging.info('Results were written to file {}')