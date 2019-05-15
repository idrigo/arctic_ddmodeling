import netCDF4 as nc
import netcdftime
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from scipy import spatial
from tqdm import tqdm

import glob

from src import cfg as cfg


def coords_to_index(filename, coords):
    # The idea is to make a table, where all the model grid points are stored in a following manner:
    # latitide - longitude - x index - y index
    # And make query to this table
    dset = nc.Dataset(filename, 'r', format="NETCDF4")

    lat = dset.variables['nav_lat'][:]
    lon = dset.variables['nav_lon'][:]

    # The following code returns 2x1d arrays for the lat-lon mesh
    lat_mesh = np.dstack(np.meshgrid(np.arange(np.shape(lat)[0]),
                                     np.arange(np.shape(lat)[1]),
                                     indexing='ij'))[:, :, 0].ravel()

    lon_mesh = np.dstack(np.meshgrid(np.arange(np.shape(lat)[0]),
                                     np.arange(np.shape(lat)[1]),
                                     indexing='ij'))[:, :, 1].ravel()
    # stack all the 1d arrays to the table
    array = np.column_stack((lat.ravel(),
                             lon.ravel(),
                             lat_mesh,
                             lon_mesh))

    latlonarr = array[:, [0, 1]]

    # Here the KD-tree algorythm is used for finding the closest spatial point (nearest neighbour)
    # More information about the algorithm
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
    # input: array(2d array of points to search), query (list-like coordinates to find neighbour)

    tree = spatial.KDTree(latlonarr, leafsize=100)
    result = tree.query(coords)[1]

    idxs = array[result][[2, 3]].astype(int)
    true_coords = array[result][[0,1]]

    return idxs, true_coords

def file_time(dset):
    tname = "time_counter"

    nctime = dset.variables[tname][:]  # get values
    t_unit = dset.variables[tname].units  # get unit

    try:
        t_cal = dset.variables[tname].calendar
    except AttributeError:  # Attribute doesn't exist
        t_cal = u"gregorian"  # or standard

    datevar = netcdftime.num2date(nctime, units=t_unit, calendar=t_cal)[0]

    datestr = datevar.strftime('%Y%m%d')
    return datestr


def extract_point(dset, point, variables_list):
    filetime = file_time(dset)
    out_list = [filetime]

    for var in variables_list:
        dimensions = dset.variables[var].dimensions

        if len(dimensions) == 4:
            data = dset.variables[var][0, 0, :, :]  # choose 2d field for one time moment on the 1st depth layer
            value = data[point[0], point[1]]

        elif len(dimensions) == 3:
            data = dset.variables[var][0, :, :]  # choose 2d field for one time moment on the 1st depth layer
            value = data[point[0], point[1]]

        out_list.append(value)

    return out_list


def makedf(filelist, varlist):
    data = []
    for f in tqdm(filelist):
        dset = nc.Dataset(f, 'r')
        try:
            filedata = extract_point(dset, idxs, varlist)
            data.append(filedata)
        except:
            tqdm.write('Problem with file {}'.format(f))
        dset.close()

    df = pd.DataFrame(data=data, columns=['date'] + varlist)
    df['Date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('Date', inplace=True)
    df.drop(['date'], axis=1, inplace=True)
    return df


lat = cfg.lat
lon = cfg.lon
path = cfg.path

idxs, true_coords = coords_to_index(glob.glob(path+'/ARCTIC_1h_T_grid_T_*.nc')[0], [lat, lon])
print('Nearest grid point is {} {}'.format(true_coords[0], true_coords[1]))

# T-grid
print('Processing T-grid files')
filelist_T = glob.glob(path+'/ARCTIC_1h_T_grid_T_*.nc')
var_list_T = ['sossheig', 'votemper', 'vosaline']

df_T = makedf(filelist_T, var_list_T)

# UV-grid
print('Processing UV-grid files')
filelist_UV = glob.glob(path+'/ARCTIC_1h_UV_grid_UV_*.nc')
var_list_UV = ['vomecrty', 'vozocrtx']

df_UV = makedf(filelist_UV, var_list_UV)

# ice-grid
print('Processing ice-grid files')
filelist_ice = glob.glob(path+'/ARCTIC_1h_ice_grid_TUV_*.nc')
var_list_ice = ['ice_volume', 'iceconc', 'icethic_cea', 'siconcat', 'sithicat', 'snowthic_cea',
                'uice_ipa', 'vice_ipa']

df_ice = makedf(filelist_ice, var_list_ice)

df_out = pd.concat([df_T, df_UV, df_ice], axis=1)
df_out.insert(loc=0,
              column='Lat',
              value=true_coords[0])
df_out.insert(loc=0,
              column='Lon',
              value=true_coords[1])

df_out.to_csv('data_{}.csv'.format(path.split('/')[-1]), sep='\t', encoding='utf-8')