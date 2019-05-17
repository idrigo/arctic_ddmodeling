import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap

import matplotlib as mpl

class Plot:

    def __init__(self):

    def mapping(data):
        m = Basemap(projection='npstere', boundinglat=50, lon_0=0, resolution='i')
        xi, yi = m(lons, lats)
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

    def plot(pred):
        cmap = plt.get_cmap('rainbow')
        cmap.set_bad('white')
        plt.title('Prediction')
        plt.imshow(pred, cmap=cmap)