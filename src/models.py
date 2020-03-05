import numpy as np

from sklearn.metrics import mean_squared_error as mse
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def init_model(model_name):
    if model_name == 'lasso':
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1, max_iter=1000)
        return model
    elif model_name =='lstm':
       pass
    else:
        raise ValueError ("Model name not found")

def predict_point(point, y_arr_train, y_arr_test, X_train, X_test, model):
    """
    Method to fit a regression on one point, given as (t, x, y)
    :param point: list or tuple of point coordinates (t, x, y)
    :return: y vector of len (t) as a regression prediction
    """

    y_train = y_arr_train[:, point[0], point[1]]
    y_test = y_arr_test[:, point[0], point[1]]

    if np.count_nonzero(~np.isnan(y_train)) == 0:  # if point is empty
        pred = np.empty_like(y_test)
        pred[:] = np.nan
    else:

        mask = ~np.isnan(X_test).any(axis=1)
        y_clean, X_train_clean = clean_data(X=X_train, y=y_train)

        X_test_clean = X_test[mask]

        model.fit(X=X_train_clean, y=y_clean)

        pred = model.predict(X_test_clean)

        pred_out = np.empty_like(y_test)
        pred_out[mask] = pred
        pred_out[~mask] = np.nan
        pred_out[pred_out < 0] = 0

        y_pred_c, y_test_c = clean_data(X=pred_out, y=y_test)
        try:
            mse_val = np.sqrt(mse(y_pred=y_pred_c, y_true=y_test_c))  # actually it is RMSE value
        except ValueError:
            mse_val = -9999

        return pred_out


def clean_data(X, y=None):
    """
    Cleans out rows with NaN from train set
    :param X:
    :param y:
    :return: cleaned data
    """
    if y is not None:
        m = np.column_stack([y, X])
        m = m[~np.isnan(m).any(axis=1)]
        y = m[:, 0]
        X = m[:, 1:]
        return y, X
    else:
        X = X[~np.isnan(X).any(axis=1)]
        return X
