import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error as mse

from src.feature_table import FeatureTable
from src.dataset import load_features, load, load_variable_years
from src.models import regress
import cfg

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 1200
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 7]

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

years_train = list(range(2010, 2012))
years_test = [2014, 2015]
X_vars = ['ice_conc', 'tair']
y_var = 'thick_cr2smos'
ft = FeatureTable(dx=2, dy=2, dt=2)
dims = np.shape(load_variable_years(y_var, years_test))

# dims = (dims[0], 10, 10) #test case
box = [0, 452, 0, 406]  # x x y y boundaries
out = np.empty(shape=dims)
out[:] = np.nan
model = Lasso(alpha=0.1, max_iter=10000)

for i in tqdm(range(box[0], box[1], 1)):
    for j in tqdm(range(box[2], box[3], 1)):
        point = (i, j)
        tqdm.write(str(point))
        y, X = load_features(X_vars=X_vars,
                             y_var=y_var,
                             years=years_train,
                             point=point,
                             feature_table=ft,
                             silent=True)
        y_test, X_test = load_features(X_vars=X_vars,
                                       y_var=y_var,
                                       years=years_test,
                                       point=point,
                                       feature_table=ft,
                                       silent=True)

        if np.count_nonzero(~np.isnan(y)) == 0:
            pred = np.empty_like(y)
            pred[:] = np.nan
            tqdm.write('passing point')
        else:
            mse_val, pred = regress(X, y, X_test, y_test, model=model)

        out[:, i, j] = pred

    np.save('data/out2.npy', out)