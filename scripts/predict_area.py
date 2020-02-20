from datetime import timedelta
import time

from sklearn.linear_model import Lasso
import logging
import numpy as np
from tqdm import tqdm

try:
    from src.main import Main
    import src.dataset as dset
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

    out = restore_array(res, indices)
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