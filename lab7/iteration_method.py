import numpy as np
from utils import check_stop


def iteration_method(phi, x1, x2, q, precision):
    x = np.asarray([x1, x2])
    while True:
        x_prev = x
        x1 = phi[0](x_prev[0], x_prev[1])
        x2 = phi[1](x_prev[0], x_prev[1])
        x = np.asarray([x1, x2])
        if q/(1-q) * check_stop(x, x_prev) < precision:
            break
    return x