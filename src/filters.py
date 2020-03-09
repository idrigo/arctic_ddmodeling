import logging

import numpy as np
from sklearn.decomposition import PCA
import scipy.ndimage
from src.utils import numpy_fillna


class MyPCA:
    def __init__(self, n_comp=5):
        self.n_comp = n_comp
        self.comps = None
    def fit_transform(self, data, exp_var=99, fit=False):

        mask = ~np.isnan(data).any(axis=1)

        data_clean = data[mask]

        if fit:
            pca = PCA()
            pca.fit(data_clean)
            var_pca = np.cumsum(pca.explained_variance_ratio_ * 100)
            self.n_comp = np.argmax(var_pca > exp_var)

        pca = PCA(n_components=self.n_comp)
        data_transformed = pca.fit_transform(data_clean)
        self.comps = np.cumsum(pca.explained_variance_ratio_ * 100)
        out = np.empty_like(data[:, :self.n_comp])
        out[mask] = data_transformed
        out[~mask] = np.nan
        return out


class Filter:

    def fit(self, data, method, window=10):
        methods = {'gaussian', 'running_mean'}

        if method not in methods:
            raise ValueError('Method must be one of {}'.format(methods))

        if method == 'gaussian':
            out = np.empty_like(data)
            for i, col in enumerate(data.T):
                out[:, i] = self.gaussian(col, window)

        if method == 'running_mean':
            out = np.empty_like(data)
            for i, col in enumerate(data.T):
                out[:, i] = self.running_mean(col, window)

        return out

    @staticmethod
    def gaussian(series, window):
        return scipy.ndimage.filters.gaussian_filter1d(series, window)

    @staticmethod
    def running_mean(series, window):
        return np.convolve(series, np.ones((window,))/window, mode='same')


class FeatureTable:
    """
    A class to create and handle feature tables
    """

    # TODO throw error if x or y not defined
    def __init__(self, x=None, y=None, data=None, t=0, dx=0, dy=0, dt=0):
        """

        :param x: x-coordinate of a point
        :param y: y-coordinate of a point
        :param data: data array
        :param t: time step
        :param dx: one-side delta of x axis
        :param dy: one-side delta of y axis
        :param dt: one-side delta of t axis
        """
        self.data = data
        self.point = [t, x, y]

        self.deltas = [dt, dy, dx]

        self.matrix = None
        self.out = None

    def select(self, data):
        """

        :return:
        """
        deltas = self.deltas
        point = self.point
        # TODO - np.clip
        idx = [[point[0] - deltas[0], point[0] + deltas[0] + 1],  # creating index matrix
               [point[1] - deltas[1], point[1] + deltas[1] + 1],
               [point[2] - deltas[2], point[2] + deltas[2] + 1]]
        idx = np.array(idx)
        for i, val in enumerate(idx):
            if val[1] > data.shape[i]:
                val[1] = data.shape[i] - 1

        idx[idx < 0] = 0
        output = data[idx[0, 0]:idx[0, 1],
                 idx[1, 0]:idx[1, 1],
                 idx[2, 0]:idx[2, 1]]

        return output.ravel()

    def gen_matrix(self, data=None, x=None, y=None, filters=None):
        """

        :param data:
        :param x:
        :param y:
        :return:
        """
        if x or y is not None:
            self.point[1] = x
            self.point[2] = y

        if data is not None:
            self.data = data

        X_out = []
        for arr in data:
            matrix = []
            for i in range(arr.shape[0]):
                self.point[0] = i
                selection = self.select(arr)
                matrix.append(selection)
            m = np.array(matrix)
            m = numpy_fillna(m)
            X_out.append(m)

        self.out = X_out
        self.apply_filter(filters=filters)
        return self.out

    def apply_filter(self, filters):
        # todo - починить пайплайн

        if filters is None:
            self.out = np.hstack([*self.out])
        else:

            if 'partial_pca' in filters:
                out = []
                if filters['partial_pca'] == 'auto':
                    for chunk in self.out:
                        m = MyPCA().fit_transform(chunk, fit=True)
                        out.append(m)
                else:
                    for chunk in self.out:

                        m = MyPCA(n_comp=filters['partial_pca']).fit_transform(chunk, fit=False)
                        out.append(m)
                self.out = np.hstack([*out])

            if 'pca' in filters:
                print('Applying PCA')
                if isinstance(self.out, list):
                    self.out = np.hstack([*self.out])
                else:
                    pass

                if filters['pca'] == 'auto':
                    self.out = MyPCA().fit_transform(self.out, fit=True)
                else:
                    self.out = MyPCA(n_comp=filters['pca']).fit_transform(self.out, fit=False)

            if 'filter_type' in filters:
                logging.debug('Applying {} filter'.format(filters['filter_type']))
                if isinstance(self.out, list):
                    self.out = np.hstack([*self.out])
                else:
                    pass
                if 'filter_window' in filters:
                    self.out = Filter().fit(data=self.out, method=filters['filter_type'], window=filters['filter_window'])
                else:
                    self.out = Filter().fit(data=self.out, method=filters['filter_type'])