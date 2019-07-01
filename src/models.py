import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error as mse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def clean_data(y, X):
    """
    Cleans out rows with NaN from train set
    :param X:
    :param y:
    :return: cleaned data
    """
    m = np.column_stack([y, X])
    m = m[~np.isnan(m).any(axis=1)]
    return m[:,0], m[:,1:]


def regress(X, y, X_test, y_test, model, mse_calc = False):
    """
    A function to apply regression and measure RMSE accuracy
    :param X: train set
    :param y: train target variable
    :param X_test: test set
    :param y_test: test target variable
    :param model: regression model to fit
    :param mse_calc:
    :return:
    """
    # todo - make a class
    y_c, X_c = clean_data(X, y)
    model.fit(X=X_c, y=y_c)

    mask = ~np.isnan(X_test).any(axis=1)
    pred = model.predict(X_test[mask])

    pred_out = np.empty_like(y_test)
    pred_out[mask] = pred
    pred_out[~mask] = np.nan
    pred_out[pred_out < 0] = 0
    y_pred_c, y_test_c = clean_data(pred_out, y_test)
    try:
        mse_val = np.sqrt(mse(y_pred=y_pred_c, y_true=y_test_c))  # actually it is RMSE value
    except ValueError:
        mse_val = -9999
    return mse_val, pred_out

