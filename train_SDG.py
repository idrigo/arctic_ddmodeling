from src.netcdftools import *

import cfg

import glob
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib

## training
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

clf = linear_model.SGDRegressor()

test_years = [2010, 2011]
train_years = list(range(1991, 2010))

test = np.array([]).reshape(0, 2)
r2 = []
mse = []


def construct_array(path):
    conc = SatelliteNC(path=path[0], variable='ice_conc')
    th = SatelliteNC(path=path[1], variable='Thickness')

    if len(conc.vector) == len(th.vector):
        data = np.column_stack((conc.vector, th.vector))
        data = data[~np.isnan(data).any(axis=1)]
        return data

    else:
        tqdm.write('Not matching lenths')
        tqdm.write('Conc: {}'.format(len(conc.vector)))
        tqdm.write('Th: {}'.format(len(th.vector)))


def plot(n_samples, r2, mse):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('number of samples')
    ax1.set_ylabel('r2 score', color=color)
    ax1.plot(n_samples, r2, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('mse', color=color)  # we already handled the x-label with ax1
    ax2.plot(n_samples, mse, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


for f in test_years:
    path = glob.glob(cfg.ice_path + '{}/*.nc'.format(f), recursive=True)

    print('Generating test set')
    data_test = construct_array(path)

    test = np.vstack([test, data_test])

y_test = test[:, 0]
X_test = test[:, 1:].reshape(-1, 1)

n_samples = [0]
for f in tqdm(train_years):
    tqdm.write(str(f))
    path = glob.glob(cfg.ice_path + '{}/*.nc'.format(f), recursive=True)
    data = construct_array(path)

    if data is not None:
        y = data[:, 0]
        X = data[:, 1].reshape(-1, 1)

        clf.partial_fit(X=X, y=y)
        _ = joblib.dump(clf, 'SDG.pkl')

        pred = clf.predict(X=X_test)
        r2.append(r2_score(y_test, pred))
        mse.append(mean_squared_error(y_test, pred))

        tqdm.write('R-2 {}'.format(r2))
        tqdm.write('MSE {}'.format(mse))

        n_samples.append(n_samples[-1]+len(y))

    else:
        tqdm.write('Passing year {}'.format(f))

plot(n_samples[1:], r2, mse)