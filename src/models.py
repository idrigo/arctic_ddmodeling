import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error as mse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.decomposition import PCA, TruncatedSVD


class Regression:
    def __init__(self, model, enable_pca=False):
        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.model = model

        self.enable_pca = enable_pca
        self.n_comp = 5

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

    def my_pca(self, data, exp_var=99.99, fit=False):

        if fit:
            #pca = PCA()
            pca = TruncatedSVD(n_components=100)
            pca.fit(data)
            var_pca = np.cumsum(pca.explained_variance_ratio_ * 100)
            self.n_comp = np.argmax(var_pca > exp_var)
            print('Number of components for {}% explained variance: {}'.format(exp_var, self.n_comp))

        #data_transformed = PCA(n_components=self.n_comp).fit_transform(data)
        data_transformed = TruncatedSVD(n_components=100).fit_transform(data)
        return data_transformed

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

        if self.enable_pca:
            X_train_clean = self.my_pca(data=X_train_clean, fit=True)
            X_test_clean = self.my_pca(data=X_test_clean, fit=False)
        self.model.fit(X=X_train_clean, y=y_clean)

        pred = self.model.predict(X_test_clean)

        pred_out = np.empty_like(y_test)
        pred_out[mask] = pred
        pred_out[~mask] = np.nan
        pred_out[pred_out < 0] = 0

        y_pred_c, y_test_c = self.clean_data(X=pred_out, y=y_test)
        try:
            mse_val = np.sqrt(mse(y_pred=y_pred_c, y_true=y_test_c))  # actually it is RMSE value
        except ValueError:
            mse_val = -9999

        print('RMSE:{}'.format(mse_val))
        return mse_val, pred_out

