import pandas as pd
import numpy as np

try:
    from src.dataset import clean_data  # for classical usage
except:
    from dataset import clean_data
from sklearn.metrics import mean_squared_error as mse

class Fitter:

    def __init__(self, train=None, test=None, y_train=None, y_test=None):

        self.train = train
        self.test = test

        self.target = None

        self.X_train = None
        self.X_test = None

        self.y_train = y_train
        self.y_test = y_test

        self.prediction = None

        self.method = None

    def define_target(self, target_var, feature_list):
        """
        Method to set train and target sets
        :param target_var: target variable
        :param feature_list: list of features to get from dataset
        :return:
        """
        if self.y_train is None:
            self.y_train = self.train[target_var]

        if self.y_test is None:
            self.y_test = self.test[target_var]

        self.y_train.name = 'Train'
        self.y_test.name = 'Test'

        #  checking if target variable in feature list provided
        if target_var in feature_list:
            feature_list.remove(target_var)

        self.X_train = self.train[feature_list]
        self.X_test = self.test[feature_list]

        return

    def predict(self):
        """
        Implements prediction using method provided
        :return:
        """

        reg = self.method
        reg.fit(self.X_train, self.y_train)
        predict = reg.predict(self.X_test)
        predict = predict.ravel()
        self.prediction = pd.Series(data=predict,
                                    index=self.y_test.index,
                                    name='Prediction')

    def r2_score(self):
        from sklearn.metrics import r2_score

        return r2_score(self.y_test, self.prediction)

    def mean_squared_error(self):
        from sklearn.metrics import mean_squared_error

        return mean_squared_error(self.y_test, self.prediction)

    def lineplot(self, title=None):
        import seaborn as sns
        import matplotlib.pyplot as plt

        data = pd.concat([self.y_test, self.prediction],
                         axis=1)
        data.columns = ['Test', 'Prediction']
        data.index.names = ['Date']

        plt.figure(figsize=(16, 6))
        sns.set(style="darkgrid")
        sns.lineplot(data=data,
                     dashes=False).set_title(title)
        return


def regress(X, y, X_test, y_test, model):
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