import logging
import time
from datetime import timedelta

import numpy as np
from tqdm import tqdm

import src.cfg as cfg
import src.data as data
import src.models as models
import src.processing as processing
import src.filters as fltr

"""
parameters = dict(years_train=list(range(2010, 2014)),
                  years_test=[2014, 2015],
                  X_vars=['ice_conc', 'icethic_cea', 'tair', 'vosaline', 'radlw'],
                  y_var='thick_cr2smos',
                  bounds=[0, 400, 0, 400],
                  step=[20, 20]
                  )
"""
parameters = dict(years_train=list(range(2010, 2012)),
                  years_test=[2014],
                  X_vars=['icethic_cea', 'vosaline', 'radlw'],
                  y_var='thick_cr2smos',
                  bounds=[0, 400, 0, 400],
                  step=[20, 20]
                  )

reg_params = dict(model='lasso',
                  dx=5,
                  dy=5,
                  dt=5
                  )
filters = dict(partial_pca=5)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)
logging.getLogger('suds').setLevel(logging.INFO)  # set INFO level for all modules

logging.info('Loading test and train data...')
y_arr_train, X_arr_train = data.load_features(parameters['y_var'],
                                              parameters['X_vars'],
                                              parameters['years_train'])

y_arr_test, X_arr_test = data.load_features(parameters['y_var'],
                                            parameters['X_vars'],
                                            parameters['years_test'])
mask = np.load(cfg.mask_path, allow_pickle=True)
logging.info('Data is loaded')

bounds = parameters['bounds']
step = parameters['step']
indices = processing.gen_indices(bounds=bounds, step=step, mask=mask)
start = time.time()
logging.info('{} points'.format(len(indices)))
logging.info('Starting iterating')

ft = fltr.FeatureTable(dx=reg_params['dx'],
                       dy=reg_params['dy'],
                       dt=reg_params['dt'])

res = []
for idx, point in tqdm(enumerate(indices), total=len(indices)):

    y_train = y_arr_train[:, point[0], point[1]]
    y_test = y_arr_test[:, point[0], point[1]]
    X_train = ft.gen_matrix(data=X_arr_train, x=point[0], y=point[1], filters=filters)
    X_test = ft.gen_matrix(data=X_arr_test, x=point[0], y=point[1], filters=filters)
    if idx == 0:
        lstm = models.MyLSTM()
        lstm.fit(X_train, y_train)
    res.append(lstm.predict(X_test))

template = np.empty_like(y_arr_test)
template[:] = np.nan
out = processing.restore_array(template=template, array_in=res, indices=indices)

for idx, val in enumerate(indices):
    (i, j) = indices[idx]
elapsed = (time.time() - start)

logging.info('Processed {} points in {} ({} points/sec)'.format(len(indices),
                                                                str(timedelta(seconds=elapsed)),
                                                                round(len(indices) / elapsed), 5))
