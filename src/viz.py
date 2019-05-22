import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import os

class Plot:

    def __init__(self, data=None):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        self.data = data
        self.lons = np.load(path+'/lons.npy')
        self.lats = np.load(path + '/lats.npy')

    def mapping(self, data=None):
        if data is None: data = self.data

        m = Basemap(projection='npstere', boundinglat=50, lon_0=0, resolution='i')
        xi, yi = m(self.lons, self.lats)
        # m.drawparallels(np.arange(-80.,81.,20.), linewidth=0.8)
        # m.drawmeridians(np.arange(-180.,181.,20.), linewidth=0.8)
        # Add Coastlines, States, and Country Boundaries
        # m.drawlsmask(land_color='c', ocean_color='w', lsmask=None)
        m.fillcontinents()
        m.drawmapboundary(fill_color='white')
        m.drawcountries()
        # cmp = cm.get_cmap('rainbow', 11)
        cmap = plt.get_cmap('rainbow')
        cmap.set_bad('white')
        m.pcolormesh(xi, yi, data, cmap=cmap, alpha=1, edgecolors='none')

        points_lons = [84,-5,162, -180]
        points_lats = [78,80,73.6,85]
        x,y=m(points_lons,points_lats)
        labels = ['1', '2', '3', '4']

        matplotlib.rcParams.update({'font.size': 22})
        for label, xpt, ypt in zip(labels, x, y):
            plt.text(xpt, ypt, label)
        m.plot(x, y, 'ro', markersize=10)
        return m

    def plot_point(self, point, data=None):
        if data is None: data=self.data
        return

    def plot(self):
        cmap = plt.get_cmap('rainbow')
        cmap.set_bad('white')
        plt.title('Prediction')
        plt.imshow(self.data, cmap=cmap)

