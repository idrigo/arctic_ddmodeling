import numpy as np


# try:
#    from src.netcdftools import MyNetCDF  # for classical usage
# except:
#    from netcdftools import MyNetCDF  # for Jupyter Notebook
try:
    from src.models import clean_data
except ModuleNotFoundError:
    from models import clean_data

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

    def select(self, data):
        """

        :return:
        """
        deltas = self.deltas
        point = self.point
        # TODO - np.clip
        idx = [[point[0] - deltas[0], point[0] + deltas[0] + 1], # creating index matrix
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

    def gen_matrix(self, data=None, x=None, y=None, enable_pca=False):
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
            for i in range(np.shape(arr)[0]):
                self.point[0] = i
                selection = self.select(arr)
                matrix.append(selection)
            m = np.array(matrix)
            X_out.append(numpy_fillna(m))

        X_out = np.hstack([*X_out])
        if enable_pca:
            self.matrix = pca(X_out)
        else:
            self.matrix = X_out
        return self.matrix


def pca(data, exp_var=95):
    # TODO make pca implementation witn n_components based on explained vatriance
    from sklearn.decomposition import PCA
    data_clean = clean_data(X=data)
    pca = PCA(100)
    pca.fit(data_clean)

    var_pca = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)
    n_comp = np.argmax(var_pca > exp_var)
    data_transformed = PCA(n_comp).fit_transform(data_clean)


    return data_transformed


def numpy_fillna(data):
    """
    Function to make 2D numpy array from array of unequal length arrays filling with nans
    :param data: input array with rows of unequal lengths
    :return: 2D numpy array
    """
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=np.float)
    out[:] = np.nan
    out[mask] = np.concatenate(data)

    return out
