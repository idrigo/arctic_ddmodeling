from sklearn.linear_model import Lasso
try:
    from src.tools import Logger, parser
    from src.main import Main

except ModuleNotFoundError:
    from tools import Logger, parser
    from main import Main

parameters = dict(years_train=list(range(2010, 2014)),
                  years_test=[2014, 2015],
                  X_vars=['ice_conc','icethic_cea', 'tair','vosaline','radlw'],
                  y_var='thick_cr2smos',
                  bounds=[0, 400, 0, 400],
                  step=[2, 2]
                  )

reg_params = dict(model=Lasso(alpha=0.1, max_iter=1000),
                  dx=5,
                  dy=5,
                  dt=5
                  )
filters = dict(partial_pca=5)


def main(parameters, reg_params):
    #indices = np.load('/Users/drigo/ITMO/_disser/surrogate/data/ice_mask_idx.npy')
    #indices = indices[26164:]
    args = parser()
    log = Logger(to_file=True, silent=False)

    m = Main(parameters=parameters, reg_params=reg_params, logger=log, filters=filters)
    m.predict_area()
    m.apply_mask()
    #m.interpolate()
    m.save()

    del m


if __name__ == '__main__':
    main(parameters, reg_params)
