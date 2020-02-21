from sklearn.linear_model import Lasso
import logging
import numpy as np

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

#indices = np.load('/Users/drigo/ITMO/_disser/surrogate/data/ice_mask_idx.npy')
#indices = indices[26164:]


if 'average' in self.par:
    self.average(self.par['average'])

logging.info('Data is loaded')

m = Main(parameters=parameters, reg_params=reg_params, filters=filters)
m.predict_area()
m.apply_mask()
#m.interpolate()
m.save()

del m


