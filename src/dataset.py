import netCDF4 as nc
import numpy as np
from scipy import interpolate

try:
    from src import cfg
except:
    import cfg


def load(variable, year):
    abspath = cfg.processed_data_path
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


def interpolation(data, method):
    """
    Fuction to restore original size of array while being calculated on reduced array
    :param data:
    :param method:
    :return:
    """
    from tqdm import tqdm

    assert len(np.shape(data)) == 3, 'Input array should be 3D'
    x = np.arange(0, data.shape[2])
    y = np.arange(0, data.shape[1])
    xx, yy = np.meshgrid(x, y)

    def interp2d(slice):
        # mask invalid values
        slice = np.ma.masked_invalid(slice)
        # get only the valid values
        x1 = xx[~slice.mask]
        y1 = yy[~slice.mask]
        newarr = slice[~slice.mask]
        GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                   (xx, yy),
                                   method=method)
        return GD1

    output = np.empty_like(data)
    for i in tqdm(range(data.shape[0])):
        try:
            output[i, :, :] = interp2d(data[i, :, :])
        except ValueError:  # TODO - разобраться что не так
            pass

    return output


# TODO сделать декоратор чтобы оборачивать интерполяции по 2d
# TODO cleanup code
def regrid(initial_data, grid_step):
    """
    Function to interpolate 2D array to the grid with different step
    :param initial_data: an initial data array
    :param grid_step: amount of cells to "merge"
    :return: reduced array
    """
    from tqdm import tqdm

    assert len(np.shape(initial_data)) == 3, 'Input array should be 3D'
    x = np.arange(0, initial_data.shape[2])
    y = np.arange(0, initial_data.shape[1])
    xx, yy = np.meshgrid(x, y)

    # new
    x1 = np.arange(0, initial_data.shape[1], grid_step)
    y1 = np.arange(0, initial_data.shape[0], grid_step)
    xx1, yy1 = np.meshgrid(x1, y1)

    def interp2d(initial_data, xx, yy, xx1, yy1):
        GD1 = interpolate.griddata(list(zip(xx.ravel(), yy.ravel())),
                                   initial_data.ravel(),
                                   (xx1, yy1), method='linear')
        return GD1
    output = np.empty((initial_data.shape[0], xx1.shape[0], xx1.shape[1]))
    for i in tqdm(range(initial_data.shape[0])):
        try:
            output[i, :, :] = interp2d(initial_data[i, :, :], xx, yy, xx1, yy1)
        except ValueError:  # TODO - разобраться что не так
            pass

    return output


def rshp(initial_data, shape):
    # todo – маска суши
    from tqdm import tqdm

    assert len(np.shape(initial_data)) == 3, 'Input array should be 3D'

    def rshp2d(data, shape):
        narr = np.pad(data, ((0, shape[0] - data.shape[0] % shape[0]), (0, shape[1] - data.shape[1] % shape[1])),
                  mode='constant',
                  constant_values=np.NaN)

        sh = shape[0], narr.shape[0] // shape[0], shape[1], narr.shape[1] // shape[1]
        return narr.reshape(sh).mean(-1).mean(1)

    output = np.empty((initial_data.shape[0], shape[0], shape[1]))
    for i in tqdm(range(initial_data.shape[0])):
        output[i, :, :] = rshp2d(initial_data[i, :, :], shape)

    return output


def iterate3d(func):
    # todo - доделать обертку
    def wrapper(data):
        from tqdm import tqdm

        assert len(np.shape(data)) == 3, 'Input array should be 3D'
        x = np.arange(0, data.shape[2])
        y = np.arange(0, data.shape[1])
        xx, yy = np.meshgrid(x, y)

        output = np.empty_like(data)
        for i in tqdm(range(data.shape[0])):
            try:
                output[i, :, :] = func(data[i, :, :])
            except ValueError:  # TODO - разобраться что не так
                pass

        return output


def mask3d(array, mask):
    """
    Args:
        array:
        mask:
    """
    array[np.isnan(array)] = 0
    mask = np.repeat(mask[None, ...], array.shape[0], axis=0)
    array = np.ma.masked_array(array, mask=mask)
    return array

