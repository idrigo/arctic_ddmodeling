import numpy as np
import pandas as pd

import os
class Preprocessing:
    """
    Class to handle csv files, generated from netcdf
    """
    def __init__(self, df=None):
        self.df = df

    def load_csv(self, filepath, continuous_check=True):
        ds = pd.read_csv(filepath, sep='\t')
        ds.set_index(pd.to_datetime(ds['Date'], format='%Y-%m-%d'), inplace=True)
        ds.drop('Date', inplace=True, axis=1)

        self.df = ds
        if continuous_check:
            self.continuous_check()

        return self.df

    def load_pickle(self, filepath, continuous_check=False):
        ds = pd.read_pickle(filepath)
        self.df = ds
        if continuous_check:
            self.continuous_check()
        return ds

    def continuous_check(self, method='ffill'):
        """
        Method to check the data for missing dates
        :param method: {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}
        :return:
        """
        idx = pd.date_range(start=self.df.index[0],
                            end=self.df.index[0] + pd.offsets.YearEnd(),
                            freq='D')
        ds = self.df.reindex(idx, fill_value=np.nan)
        print('Found {} gaps'.format(ds.isna().sum().values[0]))

        ds.fillna(method=method, inplace=True)

        self.df = ds
        return

    def velocity_module(self, x):  # simple function to convert UV velocity to velocity module
        return np.sqrt(x[0] ** 2 + x[1] ** 2)


class Logger:
    # TODO - сейчас при параллельном запуске воркеров логгирование сыпется в один файл и перекрывает друг друга.
    #  Реализовать асинхронный лог

    def __init__(self, to_file=False, silent=False):
        self.filename = None
        path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'log.txt'))
        self.logfile = path
        self.silent = silent
        self.to_file = to_file

    def start(self, parameters, reg_parameters):

        with open(self.logfile, 'a') as fo:
            fo.write('-------------------------------------------------------------------------------\n')
            fo.write(self.time)
            fo.write('\n\nPARAMETERS \n')
            for k, v in parameters.items():
                fo.write(str(k) + ' : ' + str(v) + '\n')

            fo.write('\n\nREGRESSION PARAMETERS \n')
            for k, v in reg_parameters.items():
                fo.write(str(k) + ' : ' + str(v) + '\n')

    def info(self, message):
        string = '{}\t{}'.format(self.time, message)
        if not self.silent:
            print(string)

        if self.to_file:
            with open(self.logfile, 'a') as f:
                f.write(string+'\n')
        return

    def gen_filename(self):
        return

    @property
    def time(self):
        import datetime
        time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return time_now

def parser(): # todo - доделать
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-P","--processess", metavar='P', dest="parallel",
                        help="number of processess", type=int, default=None)

    parser.add_argument("-S", "--step", dest="step",
                        help="", type=int, default=1)

    parser.add_argument("-B", "--bounds", dest="bounds",
                        help="number of processess", type=list, default=[0, 452, 0, 406])
    args = parser.parse_args()

    args.step = [args.step] * 2
    return args