import numpy as np

try:
    from src.filters import MyPCA
except ModuleNotFoundError:
    from models import MyPCA


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

        # X_out = np.hstack([*X_out])
        self.matrix = self.apply_filter(data=X_out, filters=filters)

        return self.matrix

    def apply_filter(self, data, filters):

        out = []
        if filters is None:
            out = np.hstack([*data])
            return out

        if 'partial_pca' in filters:
            if filters['partial_pca'] == 'auto':
                for chunk in data:
                    m = MyPCA().fit_transform(chunk, fit=True)
                    out.append(m)
            else:
                for chunk in data:
                    m = MyPCA(n_comp=filters['partial_pca']).fit_transform(chunk, fit=False)
                    out.append(m)
            return np.hstack([*out])

        if 'pca' in filters:
            data = np.hstack([*data])
            if filters['pca'] == 'auto':
                out = MyPCA().fit_transform(data, fit=True)
            else:
                out = MyPCA(n_comp=filters['pca']).fit_transform(data, fit=False)

            return out


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
