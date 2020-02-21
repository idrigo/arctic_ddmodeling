from datetime import timedelta
import time

from sklearn.linear_model import Lasso
import logging
import numpy as np
from tqdm import tqdm

try:
    from src.main import Main
    import src.data as dset
except ModuleNotFoundError:
    from main import Main
    import dataset as dset

parameters = dict(years_train=list(range(2010, 2014)),
                  years_test=[2014, 2015],
                  X_vars=['ice_conc','icethic_cea', 'tair','vosaline','radlw'],
                  y_var='thick_cr2smos',
                  bounds=[0, 400, 0, 400],
                  step=[2, 2]
                  )

reg_params = dict(model=Lasso(alpha=0.1, max_iter=1000),
                  dx=5,
                  dy=5,
                  dt=5
                  )
filters = dict(partial_pca=5)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)
logging.getLogger('suds').setLevel(logging.INFO)  # set INFO level for all modules

logging.info('Loading test and train data...')
y_arr_train, X_arr_train = dset.load_features(parameters['y_var'],
                                              parameters['X_vars'],
                                              parameters['years_train'])

y_arr_test, X_arr_test = dset.load_features(parameters['y_var'],
                                            parameters['X_vars'],
                                            parameters['years_test'])

logging.info('Data is loaded')


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

    out = dset.restore_array(res, indices)
    coeff_len = next(len(item) for item in coeffs if item is not None)
    coeffcients_out = np.empty((coeff_len, np.shape(self.out)[1], np.shape(self.out)[2]))
    coeffcients_out[:] = np.nan

    for idx, val in enumerate(indices):
        (i, j) = indices[idx]
        self.coeffcients_out[:, i, j] = coeffs[idx]
    elapsed = (time.time() - start)

    logging.info('Processed {} points in {} ({} points/sec)'.format(len(indices),
                                                                    str(timedelta(seconds=elapsed)),
                                                                    round(len(indices) / elapsed), 5))
    return out
