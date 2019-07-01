import numpy as np


# try:
#    from src.netcdftools import MyNetCDF  # for classical usage
# except:
#    from netcdftools import MyNetCDF  # for Jupyter Notebook


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

    def select(self):
        """

        :return:
        """
        data = self.data
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

    def gen_matrix(self, data=None, x=None, y=None):
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

        matrix = []
        # TODO: do smth if rows are not equal length (for time and bounds)
        for i in range(np.shape(self.data)[0]):
            self.point[0] = i
            selection = self.select()
            matrix.append(selection)
        m = np.array(matrix)

        self.matrix = numpy_fillna(m)
        return self.matrix


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
