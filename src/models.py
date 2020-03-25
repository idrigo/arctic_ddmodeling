import numpy as np

import warnings
from sklearn.linear_model import Lasso

warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import backend as K


class MyLasso:
    def __init__(self):
        self.model = Lasso(alpha=0.01, max_iter=1000)

        return

    def fit(self, X_train, y_train):
        y_clean, X_train_clean = clean_data(X=X_train, y=y_train)
        self.model.fit(X=X_train_clean, y=y_clean)

    def predict(self, X_test):
        mask = ~np.isnan(X_test).any(axis=1)
        pred_out = np.empty((X_test.shape[0]))

        X_test = X_test[mask]

        pred = self.model.predict(X_test).ravel()
        pred_out[mask] = pred
        pred_out[~mask] = np.nan
        pred_out[pred_out < 0] = 0
        return pred_out


def reshape2d(array):
    out = array.reshape((array.shape[0], 1, array.shape[1]))
    return out


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

class MyLSTM:
    def __init__(self):
        self.history = None
        self.model = None

    def fit(self, X_train, y_train, parameters=None):
        y_train, X_train = clean_data(X=X_train, y=y_train)
        X_train = reshape2d(X_train)

        self.model = Sequential()
        if not parameters:
            parameters = dict(n_neurons=20,
                              epochs=70,
                              batch_size=30,
                              loss='mae')

        self.model.add(LSTM(parameters['n_neurons'],
                            input_shape=(X_train.shape[1], X_train.shape[2]),
                            return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(parameters['n_neurons'], return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(parameters['n_neurons']))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))

        self.model.compile(loss=parameters['loss'], optimizer='adam')
        # print(self.model.summary())
        self.history = self.model.fit(X_train, y_train,
                                      epochs=parameters['epochs'],
                                      batch_size=parameters['batch_size'],
                                      verbose=0, shuffle=False)

    def predict(self, X_test):
        mask = ~np.isnan(X_test).any(axis=1)
        pred_out = np.empty((X_test.shape[0]))

        X_test = X_test[mask]
        X_test = reshape2d(X_test)

        pred = self.model.predict(X_test).ravel()
        pred_out[mask] = pred
        pred_out[~mask] = np.nan
        pred_out[pred_out < 0] = 0
        return pred_out


def predict_point(y_train, y_test, X_train, X_test, model):
    """
    Method to fit a regression on one point, given as (t, x, y)
    :return: y vector of len (t) as a regression prediction
    """

    if np.count_nonzero(~np.isnan(y_train)) == 0:  # if point is empty
        pred = np.empty_like(y_test)
        pred[:] = np.nan
    else:
        if model == 'lasso':
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
        elif model == 'lstm':
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
