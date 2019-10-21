import numpy as np
from sklearn.decomposition import PCA
import scipy.ndimage

class MyPCA:
    def __init__(self, n_comp=5):
        self.n_comp = n_comp

    def fit_transform(self, data, exp_var=99, fit=False):

        mask = ~np.isnan(data).any(axis=1)


        data_clean = data[mask]

        if fit:
            pca = PCA()
            pca.fit(data_clean)
            var_pca = np.cumsum(pca.explained_variance_ratio_ * 100)
            self.n_comp = np.argmax(var_pca > exp_var)
            print('Number of components for {}% explained variance: {}'.format(exp_var, self.n_comp))

        data_transformed = PCA(n_components=self.n_comp).fit_transform(data_clean)
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

