import numpy as np
import src.dataset as dset
from src.feature_table import FeatureTable
from sklearn.linear_model import Lasso
from src.models import regress



parameters = dict(years_train=list(range(2010, 2012)),
                  years_test=[2014, 2015],
                  X_vars=['ice_conc', 'tair'],
                  y_var='thick_cr2smos',
                  boundaries=[0, 200, 0, 200],  # N-S-W-E
                  parallelization=16)

reg_params = dict(model=Lasso(alpha=0.1, max_iter=10000),
                  dx=2,
                  dy=2,
                  dt=2
                  )



class Main:
    def __init__(self, parameters, reg_params):

        self.par = parameters
        self.reg_params = reg_params

        self.y_arr_train, self.X_arr_train = dset.load_features(self.par['y_var'],
                                                                self.par['X_vars'],
                                                                self.par['years_train'])

        self.y_arr_test, self.X_arr_test = dset.load_features(self.par['y_var'],
                                                              self.par['X_vars'],
                                                              self.par['years_test'])

        self.dimensions = np.shape(dset.load_variable_years(self.par['y_var'],
                                                            self.par['years_test']))

        self.ft = FeatureTable(dx=reg_params['dx'],
                               dy=reg_params['dy'],
                               dt=reg_params['dt'])
        self.model = reg_params['model']

        out = np.empty(shape=self.dimensions)
        out[:] = np.nan
        self.out = out

    def predict_point(self, point):

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

    def predict_area(self):
        from multiprocessing import Pool
        from itertools import product

        if self.par['parallelization']:
            with Pool(64) as pool:
                i = range(self.par['boundaries'][2], self.par['boundaries'][3])
                j = range(self.par['boundaries'][0], self.par['boundaries'][1])
                out_flat = list(product(i, j))
                res = pool.starmap(self.predict_point, out_flat, 1)
                #TODO processing of thi output (reshaping)

        else:
            pass #TODO - serial implementation