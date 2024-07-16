import numpy as np

def log_ratio(y: np.ndarray, y_hat: np.ndarray):
    """
    The log ratio is a symmetric and relative accuracy score (Tofallis, 2015)
    """
    # compute quotient
    log_quotient    = np.log(y_hat/y)

    # mean squared residual
    mean_error      = np.mean(np.square(log_quotient))

    # return error
    return mean_error