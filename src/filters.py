import numpy as np
from sklearn.decomposition import PCA


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

    def __init__(self):
        pass

    def gaussian(self):
        return