import numpy as np

import warnings
from sklearn.linear_model import Lasso
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def predict_point(y_train, y_test, X_train, X_test, model):
    """
    Method to fit a regression on one point, given as (t, x, y)
    :param point: list or tuple of point coordinates (t, x, y)
    :return: y vector of len (t) as a regression prediction
    """


    if np.count_nonzero(~np.isnan(y_train)) == 0:  # if point is empty
        pred = np.empty_like(y_test)
        pred[:] = np.nan
    else:
        if model=='lasso':
            model = Lasso(alpha=0.1, max_iter=1000)

            y_clean, X_train_clean = clean_data(X=X_train, y=y_train)

            model.fit(X=X_train_clean, y=y_clean)
            mask = ~np.isnan(X_test).any(axis=1)
            X_test_clean = X_test[mask]

            pred = model.predict(X_test_clean)

            pred_out = np.empty_like(y_test)
            pred_out[mask] = pred
            pred_out[~mask] = np.nan
            pred_out[pred_out < 0] = 0

            return pred_out
        elif model=='lstm':
            model = Sequential()
            model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
            model.add(Dense(1))
            model.compile(loss='mae', optimizer='adam')

def clean_data(X, y=None):
    """
    Cleans out rows with NaN from train set
    :param X:
    :param y:
    :return: cleaned data
    """
    if y is not None:
        m = np.column_stack([y, X])
        mask = ~np.isnan(m).any(axis=1)
        m = m[mask]
        y = m[:, 0]
        X = m[:, 1:]
        return y, X
    else:
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        return X
