import numpy as np
from sklearn.linear_model import Lasso
from datetime import timedelta
import time
from tqdm import tqdm

try:
    import src.dataset as dset
    from src.feature_table import FeatureTable
    from src.models import regress
    import src.cfg as cfg

except ModuleNotFoundError:
    import sys
    import os
    # abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    # sys.path.append('/home/hpc-rosneft/drigo/surrogate/src/')
    import cfg
    import dataset as dset
    from feature_table import FeatureTable
    from models import regress

parameters = dict(years_train=list(range(2010, 2012)),
                  years_test=[2014, 2015],
                  X_vars=['ice_conc', 'tair'],
                  y_var='thick_cr2smos'
                  )

reg_params = dict(model=Lasso(alpha=0.1, max_iter=10000),
                  dx=2,
                  dy=2,
                  dt=2
                  )


class Main:

    def __init__(self, parameters=parameters, reg_params=reg_params, silent=False, log=False):

        self.par = parameters
        self.reg_params = reg_params

        self.silent = silent
        self.log = log
        self.mask = np.load(cfg.mask_path)

        self.y_arr_train = None
        self.X_arr_train = None

        self.y_arr_test = None
        self.X_arr_test = None

        self.ft = None
        self.out = None

        self.model = reg_params['model']

        self.init_data()

    def init_data(self):
        """
        Method to define class me
        :return:
        """
        self.yell('Loading test and train data...')

        self.y_arr_train, self.X_arr_train = dset.load_features(self.par['y_var'],
                                                                self.par['X_vars'],
                                                                self.par['years_train'])

        self.y_arr_test, self.X_arr_test = dset.load_features(self.par['y_var'],
                                                              self.par['X_vars'],
                                                              self.par['years_test'])

        self.ft = FeatureTable(dx=reg_params['dx'],
                               dy=reg_params['dy'],
                               dt=reg_params['dt'])
        out = np.empty_like(self.y_arr_test)
        out[:] = np.nan
        self.out = out
        self.yell('Data is loaded')

    def predict_point(self, point):
        """

        :param point:
        :return:
        """
        y_train = self.y_arr_train[:, point[0], point[1]]
        y_test = self.y_arr_test[:, point[0], point[1]]

        X_train = []
        X_test = []
        for var_n, var in enumerate(self.par['X_vars']):
            X = self.ft.gen_matrix(data=self.X_arr_train[var_n], x=point[0], y=point[1])
            X_train.append(X)
            X = self.ft.gen_matrix(data=self.X_arr_test[var_n], x=point[0], y=point[1])
            X_test.append(X)

        X_train = np.hstack([*X_train])
        X_test = np.hstack([*X_test])

        if np.count_nonzero(~np.isnan(y_train)) == 0:
            pred = np.empty_like(y_train)
            pred[:] = np.nan
        else:
            mse_val, pred = regress(X_train, y_train, X_test, y_test, model=reg_params['model'])

        return pred

    def predict_area(self, bounds, step, parallel=None):
        from multiprocessing import Pool

        indices = self.gen_indices(bounds, step)
        start = time.time()
        self.yell('{} points'.format(len(indices)))
        if parallel:
            assert type(parallel) is int, 'Number of processes should be int type'

            self.yell('Starting regression using {} cores'.format(parallel))
            with Pool(parallel) as pool:
                res = pool.starmap(self.predict_point, indices, 1)

            '''
            res = res.reshape(dims[1], dims[2], 730)
            res = np.swapaxes(res, 2, 0)
            np.save('res.npy', res)
            '''

        else:
            self.yell('Starting regression on a single core')
            res = []
            for idx, val in tqdm(enumerate(indices), total=len(indices)):
                res.append(self.predict_point(val))

        res = np.array(res)
        result = self.restore_array(res, indices)
        elapsed = (time.time() - start)

        self.yell('Processed {} points in {} ({} points/sec)'.format(len(indices),
                                                                     str(timedelta(seconds=elapsed)),
                                                                     round(len(indices) / elapsed), 5))
        return result

    def gen_indices(self, bounds, step):
        from itertools import product
        i = range(bounds[0],
                  bounds[1],
                  step[0])

        j = range(bounds[2],
                  bounds[3],
                  step[1])

        indices = list(product(i, j))
        return indices

    def restore_array(self, array_in, indices):

        self.yell('Constructing output array')
        for idx, val in enumerate(indices):
            (i, j) = indices[idx]
            self.out[:, i, j] = array_in[idx]

        return self.out

    def post_process(self, array):
        array[np.isnan(array)] = 0
        array = np.ma.masked_array(array, mask=self.mask)
        return array

    def interpolation(self, data, method='nearest'):
        from scipy import interpolate
        self.yell('Interpolating data using {} method'.format(method))
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

        output = np.apply_along_axis(interp2d, 0, data)
        return output

    def logging(self):
        # TODO - логгирование
        return

    def yell(self, message):
        if not self.silent:
            print(message)
        else:
            pass

    def save(self):
        return


if __name__ == '__main__':
    m = Main(parameters=parameters)
    o = m.predict_area(bounds=[100, 200, 100, 200], step=[2, 2], parallel=None)
    # np.save('res.npy', 0)
