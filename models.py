import pandas as pd
import numpy as np


class Fitter:

    def __init__(self, train, test):

        self.train = train
        self.test = test

        self.target = None

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.prediction = None


    def define_target(self, target_var, feature_list):

        self.y_train = self.train[target_var]
        self.y_test = self.test[target_var]

        #  checking if target variable in feature list provided
        if target_var in feature_list:
            feature_list.remove(target_var)

        self.X_train = self.train[feature_list]
        self.X_test = self.test[feature_list]

        return

    def linear_regression_prediction(self):

        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(self.X_train, self.y_train)
        predict = reg.predict(self.X_test)
        self.prediction = pd.Series(data=predict,
                                    index=self.y_test.index,
                                    name = 'Prediction')
        return self.prediction

    def r2_score(self):
        from sklearn.metrics import r2_score

        return r2_score(self.y_test, self.prediction)

    def mean_squared_error(self):
        from sklearn.metrics import mean_squared_error

        return mean_squared_error(self.y_test, self.prediction)

    def plot(self):
        import matplotlib.pyplot as plt
        import seaborn as sns



        return