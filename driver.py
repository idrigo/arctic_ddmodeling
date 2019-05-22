from sklearn.linear_model import Lasso
import os

try:
    from src.tools import Logger, parser
    from src.main import Main

except ModuleNotFoundError:
    from tools import Logger, parser
    from main import Main

parameters = dict(years_train=list(range(2010, 2014)),
                  years_test=[2014, 2015],
                  X_vars=['ice_conc', 'tair', 'votemper'],
                  y_var='thick_cr2smos',
                  bounds=[0, 452, 0, 406],
                  step=[5, 5]
                  )

reg_params = dict(model=Lasso(alpha=0.1, max_iter=10000),
                  dx=5,
                  dy=5,
                  dt=5
                  )


def main(parameters, reg_params):
    args = parser()
    log = Logger(to_file=True, silent=False)

    m = Main(parameters=parameters, reg_params=reg_params, logger=log)
    m.predict_area()
    m.interpolate()
    m.apply_mask()
    m.save()

    del m


if __name__ == '__main__':
    main(parameters, reg_params)
