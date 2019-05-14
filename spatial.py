import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from multiprocessing import Pool
from timeit import default_timer as timer
from itertools import product

from sklearn.linear_model import Lasso
from src.feature_table import FeatureTable
from src.dataset import load_features, load, load_variable_years
from src.models import regress


def predict(i,j):
    point = (i,j)
    y_train = y_arr_train[:, point[0], point[1]]
    y_test = y_arr_test[:, point[0], point[1]]

    X_train = []
    X_test = []

    for var_n, var in enumerate(cfg['X_vars']):
        X = cfg['ft'].gen_matrix(data=X_arr_train[var_n], x=point[0], y=point[1])
        X_train.append(X)
        X = cfg['ft'].gen_matrix(data=X_arr_test[var_n], x=point[0], y=point[1])
        X_test.append(X)

    X_train = np.hstack([*X_train])
    X_test = np.hstack([*X_test])

    if np.count_nonzero(~np.isnan(y_train)) == 0:
        pred = np.empty_like(y_train)
        pred[:] = np.nan
    else:
        mse_val, pred = regress(X_train, y_train, X_test, y_test, model=cfg['model'])

    return pred


def idx(i, j):
    return [i, j]


if __name__ == '__main__':
    # configuration
    cfg = dict(years_train=list(range(2010, 2012)),
               years_test=[2014, 2015],
               X_vars=['ice_conc', 'tair'],
               y_var='thick_cr2smos',
               ft=FeatureTable(dx=2, dy=2, dt=2),
               model=Lasso(alpha=0.1, max_iter=10000),
               boundaries=[0, 200, 0, 200]  # N-S-W-E
               )

    dims = np.shape(load_variable_years(cfg['y_var'], cfg['years_test']))

    out = np.empty(shape=dims)
    out[:] = np.nan

    y_arr_train, X_arr_train = load_features(cfg['y_var'], cfg['X_vars'], cfg['years_train'])
    y_arr_test, X_arr_test = load_features(cfg['y_var'], cfg['X_vars'], cfg['years_test'])

    # i = range(box[0], box[1], 1)
    # j = range(box[2], box[3], 1)

    i = range(dims[1])
    j = range(dims[2])

    out_flat = list(product(i, j))

    start = timer()
    with Pool(64) as pool:
        res = pool.starmap(predict, out_flat, 1)
        # idx = pool.starmap(idx, out_flat)
    np.save('res.npy', res)
    # np.save('idx.npy', idx)
    end = timer()
    print('Time = ', end - start)
