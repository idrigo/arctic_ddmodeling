import logging
import numpy as np
from scipy.interpolate import interpolate


def gen_indices(bounds, step, mask):
    """
    :param bounds:
    :param step:
    :return:
    """
    from itertools import product
    i = range(bounds[0],
              bounds[1],
              step[0])

    j = range(bounds[2],
              bounds[3],
              step[1])

    idx = list(product(i, j))
    idx[:] = [tup for tup in idx if mask[tup] == False]
    return idx

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
    output = np.empty(( initial_data.shape[0], xx1.shape[0], xx1.shape[1]))
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
                  constant_values=np.nan)

        sh = shape[0], narr.shape[0] // shape[0], shape[1], narr.shape[1] // shape[1]
        return narr.reshape(sh).mean(-1).mean(1)

    output = np.empty((initial_data.shape[0], shape[0], shape[1]))
    output[:] = np.nan
    for i in tqdm(range(initial_data.shape[0])):
        output[i, :, :] = rshp2d(initial_data[i, :, :], shape)

    return output


def restore_array(array_in, indices):
    logging.info('Constructing output array')
    for idx, val in enumerate(indices):
        (i, j) = indices[idx]
        out[:, i, j] = array_in[idx]

    return out


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