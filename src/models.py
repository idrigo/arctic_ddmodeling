import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error as mse
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)

def clean_data(X, y):
    m = np.column_stack([y, X])
    m = m[~np.isnan(m).any(axis=1)]
    return m[:,0], m[:,1:]


def regress(X, y, X_test, y_test, model, mse_calc = False):
    # todo - make a class
    reg = model
    y_c, X_c = clean_data(X, y)
    reg.fit(X=X_c, y=y_c)

    mask = ~np.isnan(X_test).any(axis=1)
    pred = reg.predict(X_test[mask])
    # idx = np.argwhere(mask).ravel()

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