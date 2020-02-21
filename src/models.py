import numpy as np

from sklearn.metrics import mean_squared_error as mse
import warnings

try:
    from src.filters import MyPCA
except ModuleNotFoundError:
    from filters import MyPCA

warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def predict_point(point, y_arr_train, y_arr_test, X_arr_train , X_arr_test):
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
        X_train = ft.gen_matrix(data=X_arr_train, x=point[0], y=point[1], filters=filters)
        X_test = ft.gen_matrix(data=X_arr_test, x=point[0], y=point[1], filters=filters)

        regression = Regression(model=self.reg_params['model'])
        mse_val, pred = regression.regress(X_train=X_train, y_train=y_train,
                                           X_test=X_test, y_test=y_test)
        coeff = regression.model.coef_

    return pred

class Regression:
    def __init__(self, model):
        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.model = model

    def regress(self, X_train, y_train, X_test, y_test):
        """
        A function to apply regression and measure RMSE accuracy
        :param X_train: train set
        :param y_train: train target variable
        :param X_test: test set
        :param y_test: test target variable
        :return:
        """

        mask = ~np.isnan(X_test).any(axis=1)
        y_clean, X_train_clean = self.clean_data(X=X_train, y=y_train)

        X_test_clean = X_test[mask]

        self.model.fit(X=X_train_clean, y=y_clean)

        pred = self.model.predict(X_test_clean)

        pred_out = np.empty_like(y_test)
        pred_out[mask] = pred
        pred_out[~mask] = np.nan
        pred_out[pred_out < 0] = 0

        y_pred_c, y_test_c = clean_data(X=pred_out, y=y_test)
        try:
            mse_val = np.sqrt(mse(y_pred=y_pred_c, y_true=y_test_c))  # actually it is RMSE value
        except ValueError:
            mse_val = -9999

        return mse_val, pred_out

def clean_data(self, X, y=None):
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