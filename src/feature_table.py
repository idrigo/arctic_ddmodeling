import numpy as np
import cfg

try:
    from src.netcdftools import MyNetCDF  # for classical usage
except:
    from netcdftools import MyNetCDF  # for Jupyter Notebook


class FeatureTable:
    """
    A class to create and handle feature tables –
    """

    # TODO throw error if x or y not defined
    def __init__(self, x=None, y=None, data=None, t=0, dx=0, dy=0, dt=0, autoreg=None):

        self.data = data
        self.selection = None
        self.point = [t, x, y]

        self.dx = dx
        self.dy = dy
        self.dt = dt

        self.matrix = None
        self.autoreg = autoreg

    def field_idx(self):

        deltas = [self.dt, self.dx, self.dy]
        d = []

        for i in range(3):
            start = self.point[i] - deltas[i]
            stop = self.point[i] + deltas[i] + 1
            l = list(range(start, stop))
            d.append(l)

        mesh = np.meshgrid(*d)
        out = []
        for arr in mesh:
            out.append(arr.ravel())

        idx = np.column_stack([*out])

        datashape = np.shape(self.data)
        # sel = (idx >= 0).all(axis=1)

        idx = idx[(idx[:, 0] >= 0) & (idx[:, 0] < datashape[0])
                  & (idx[:, 1] >= 0) & (idx[:, 1] < datashape[1])
                  & (idx[:, 2] >= 0) & (idx[:, 2] < datashape[2])]
        '''
        idx = idx[(idx[:, 1] >= 0) & (idx[:, 1] < datashape[1])
                  & (idx[:, 2] >= 0) & (idx[:, 2] < datashape[2])]
        '''
        return idx

    def select(self):
        selection = []
        indexes = self.field_idx()

        for i, val in enumerate(indexes):
            ix = tuple(indexes[i])
            selection.append(self.data[ix])

        self.selection = np.array(selection)
        return self.selection

    def gen_matrix(self, data=None, x=None, y=None, autoreg=None):
        if autoreg is not None:
            self.autoreg = autoreg

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
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=np.float)
    out[:] = np.nan
    out[mask] = np.concatenate(data)

    return out


