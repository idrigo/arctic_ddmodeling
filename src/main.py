import numpy as np
from datetime import timedelta
import time
from tqdm import tqdm
import logging

try:
    import src.dataset as dset
    from src.feature_table import FeatureTable
    import src.cfg as cfg
    from src.models import Regression
except ModuleNotFoundError:
    import sys
    import os
    import cfg
    import dataset as dset
    from feature_table import FeatureTable
    from models import Regression


def init_data(parameters, reg_params):
    """

    :return:
    """
    logging.info('Loading test and train data...')
    y_arr_train, X_arr_train = dset.load_features(parameters['y_var'],
                                                  parameters['X_vars'],
                                                  parameters['years_train'])

    y_arr_test, X_arr_test = dset.load_features(parameters['y_var'],
                                                parameters['X_vars'],
                                                parameters['years_test'])

    logging.info('Data is loaded')

    return y_arr_train, y_arr_test, X_arr_train , X_arr_test


def predict_point(point, y_arr_train, y_arr_test, X_arr_train , X_arr_test):
    """
    Method to fit a regression on one point, given as (t, x, y)
    :param point: list or tuple of point coordinates (t, x, y)
    :return: y vector of len (t) as a regression prediction
    """

    y_train = y_arr_train[:, point[0], point[1]]
    y_test = y_arr_test[:, point[0], point[1]]

    if np.count_nonzero(~np.isnan(y_train)) == 0:  # if point is empty
        pred = np.empty_like(y_test)
        pred[:] = np.nan
    else:
        X_train = ft.gen_matrix(data=X_arr_train, x=point[0], y=point[1], filters=filters)
        X_test = ft.gen_matrix(data=X_arr_test, x=point[0], y=point[1], filters=filters)

        regression = Regression(model=self.reg_params['model'])
        mse_val, pred = regression.regress(X_train=X_train, y_train=y_train,
                                           X_test=X_test, y_test=y_test)
        coeff = regression.model.coef_

    return pred


class Main:

    def __init__(self, parameters, reg_params, filters):
        """

        :param parameters:
        :param reg_params:
        :param logger:
        """

        self.par = parameters
        self.reg_params = reg_params
        self.filters = filters

        self.mask = np.load(cfg.mask_path)

        self.y_arr_train = None
        self.X_arr_train = None

        self.y_arr_test = None
        self.X_arr_test = None

        self.ft = None
        self.out = None

        self.coeff = None
        self.coeffcients_out = None
        self.init_data()

    def init_data(self):
        """

        :return:
        """
        logging.info('Loading test and train data...')

        self.y_arr_train, self.X_arr_train = dset.load_features(self.par['y_var'],
                                                                self.par['X_vars'],
                                                                self.par['years_train'])

        self.y_arr_test, self.X_arr_test = dset.load_features(self.par['y_var'],
                                                              self.par['X_vars'],
                                                              self.par['years_test'])

        self.ft = FeatureTable(dx=self.reg_params['dx'],
                               dy=self.reg_params['dy'],
                               dt=self.reg_params['dt'])
        out = np.empty_like(self.y_arr_test)
        out[:] = np.nan
        self.out = out

        logging.info('Data is loaded')

    def predict_point(self, point):
        """
        Method to fit a regression on one point, given as (t, x, y)
        :param point: list or tuple of point coordinates (t, x, y)
        :return: y vector of len (t) as a regression prediction
        """

        y_train = self.y_arr_train[:, point[0], point[1]]
        y_test = self.y_arr_test[:, point[0], point[1]]

        if np.count_nonzero(~np.isnan(y_train)) == 0:  # if point is empty
            pred = np.empty_like(y_test)
            pred[:] = np.nan
        else:
            X_train = self.ft.gen_matrix(data=self.X_arr_train, x=point[0], y=point[1], filters=self.filters)
            X_test = self.ft.gen_matrix(data=self.X_arr_test, x=point[0], y=point[1], filters=self.filters)

            regression = Regression(model=self.reg_params['model'])
            mse_val, pred = regression.regress(X_train=X_train, y_train=y_train,
                                               X_test=X_test, y_test=y_test)
            self.coeff = regression.model.coef_

        return pred

    def predict_area(self, bounds=None, step=None, indices=None):
        """

        :param bounds:
        :param step:
        :param parallel:
        :return:
        """

        if bounds is None: bounds = self.par['bounds']

        if step is None: step = self.par['step']

        if indices is None: indices = self.gen_indices(bounds, step)
        start = time.time()
        logging.info('{} points'.format(len(indices)))
        logging.info('Starting regression on a single core')

        res = []
        coeffs = []
        for idx, point in tqdm(enumerate(indices), total=len(indices)):
            res.append(self.predict_point(point))
            coeffs.append(self.coeff)

        self.out = self.restore_array(res, indices)
        coeff_len = next(len(item) for item in coeffs if item is not None)
        self.coeffcients_out = np.empty((coeff_len, np.shape(self.out)[1], np.shape(self.out)[2]))
        self.coeffcients_out[:] = np.nan

        for idx, val in enumerate(indices):
            (i, j) = indices[idx]
            self.coeffcients_out[:, i, j] = coeffs[idx]
        elapsed = (time.time() - start)

        logging.info('Processed {} points in {} ({} points/sec)'.format(len(indices),
                                                                         str(timedelta(seconds=elapsed)),
                                                                         round(len(indices) / elapsed), 5))
        return self.out

    def gen_indices(self, bounds, step):
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
        idx[:] = [tup for tup in idx if self.mask[tup] == False]
        return idx

    def restore_array(self, array_in, indices):

        logging.info('Constructing output array')
        for idx, val in enumerate(indices):
            (i, j) = indices[idx]
            self.out[:, i, j] = array_in[idx]

        return self.out

    def apply_mask(self, array=None):
        if array is None:
            array = self.out

        self.out = dset.mask3d(array=array, mask=self.mask)
        self.coeffcients_out = dset.mask3d(array=self.coeffcients_out, mask=self.mask)
        return self.out

    def interpolate(self, method='nearest'):

        logging.info('Interpolating data using {} method'.format(method))

        self.out = dset.interpolation(data=self.out, method=method)
        self.coeffcients_out = dset.interpolation(data=self.coeffcients_out, method=method)
        return self.out

    def save(self):  # todo - доделать
        import datetime
        import os
        time_now = datetime.datetime.now().strftime("%m%d_%H%M")
        logging.info('Saving output to file')

        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'output'))
        self.out.dump(os.path.join(path, 'res_{}.npy'.format(time_now)))

        self.coeffcients_out.dump((os.path.join(path, 'coeffs_{}.npy'.format(time_now))))
        logging.info('Results were written to file {}')

    def get_ft(self, point):

        X_train = self.ft.gen_matrix(data=self.X_arr_train, x=point[0], y=point[1], filters=self.filters)
        X_test = self.ft.gen_matrix(data=self.X_arr_test, x=point[0], y=point[1], filters=self.filters)

        return X_train, X_test
