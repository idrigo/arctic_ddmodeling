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
    from src.tools import Logger, parser
except ModuleNotFoundError:
    import sys
    import os
    # abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    # sys.path.append('/home/hpc-rosneft/drigo/surrogate/src/')
    import cfg
    import dataset as dset
    from feature_table import FeatureTable
    from models import regress
    from tools import Logger, parser

parameters = dict(years_train=list(range(2010, 2012)),
                  years_test=[2014, 2015],
                  X_vars=['ice_conc', 'tair', 'votemper'],
                  y_var='thick_cr2smos',
                  bounds=[0, 400, 0, 400],
                  step=[1, 1]
                  )

reg_params = dict(model=Lasso(alpha=0.1, max_iter=10000),
                  dx=2,
                  dy=2,
                  dt=2
                  )
log = Logger(to_file=False, silent=True)


class Main:
    def __init__(self, parameters=parameters, reg_params=reg_params, logger=log):

        self.log = logger
        self.par = parameters
        self.reg_params = reg_params
        self.log.start(parameters, reg_params)

        self.mask = np.load(cfg.mask_path)

        self.y_arr_train = None
        self.X_arr_train = None

        self.y_arr_test = None
        self.X_arr_test = None

        self.ft = None
        self.out = None

        self.init_data()



    def init_data(self):
        """
        Method to define class arguments
        :return:
        """
        self.log.info('Loading test and train data...')

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
        self.log.info('Data is loaded')

    def predict_point(self, point):
        """
        Method to fit a regression on one point, given as (t, x, y)
        :param point: list or tuple of point coordinates (t, x, y)
        :return: y vector of len (t) as a regression prediction
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
            pred = np.empty_like(y_test)
            pred[:] = np.nan
        else:
            mse_val, pred = regress(X_train, y_train, X_test, y_test, model=self.reg_params['model'])

        return pred

    def predict_area(self, bounds=None, step=None, parallel=None):

        if bounds is None: bounds = self.par['bounds']

        if step is None: step = self.par['step']

        indices = self.gen_indices(bounds, step)
        start = time.time()
        self.log.info('{} points'.format(len(indices)))
        if parallel:
            from multiprocessing import Pool
            # TODO - разобраться что тут не так
            assert type(parallel) is int, 'Number of processes should be int type'

            self.log.info('Starting regression using {} cores'.format(parallel))
            with Pool(parallel) as pool:
                res = pool.starmap(self.predict_point, indices, 1)

            '''
            res = res.reshape(dims[1], dims[2], 730)
            res = np.swapaxes(res, 2, 0)
            np.save('res.npy', res)
            '''

        else:
            self.log.info('Starting regression on a single core')
            res = []
            for idx, point in tqdm(enumerate(indices), total=len(indices)):
                res.append(self.predict_point(point))

        result = self.restore_array(res, indices)
        elapsed = (time.time() - start)

        self.log.info('Processed {} points in {} ({} points/sec)'.format(len(indices),
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

        idx = list(product(i, j))
        idx[:] = [tup for tup in idx if self.mask[tup] == False]
        return idx

    def restore_array(self, array_in, indices):

        self.log.info('Constructing output array')
        for idx, val in enumerate(indices):
            (i, j) = indices[idx]
            self.out[:, i, j] = array_in[idx]

        return self.out

    def apply_mask(self, array=None):
        if array is None:
            array = self.out

        self.out = dset.mask3d(array=array, mask=self.mask)
        return self.out

    def interpolate(self, data=None, method='nearest'):

        log.info('Interpolating data using {} method'.format(method))
        if data is None:
            data = self.out

        self.out = dset.interpolate(data=data, method=method)
        return self.out

    def save(self):  # todo - доделать
        import datetime
        import os
        time_now = datetime.datetime.now().strftime("%m%d_%H%M")
        self.log.info('Saving results to file')
        fname = 'res_{}.npy'.format(time_now)
        fname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'results', fname))
        self.out.dump(fname)
        self.log.info('Results were written to file {}'.format(fname))

    def get_ft(self, point):
        X_train = []
        X_test = []
        for var_n, var in enumerate(self.par['X_vars']):
            X = self.ft.gen_matrix(data=self.X_arr_train[var_n], x=point[0], y=point[1])
            X_train.append(X)
            X = self.ft.gen_matrix(data=self.X_arr_test[var_n], x=point[0], y=point[1])
            X_test.append(X)

        X_train = np.hstack([*X_train])
        X_test = np.hstack([*X_test])
        return X_train