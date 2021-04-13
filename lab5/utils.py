import numpy as np
from math import floor, fabs, ceil, log10
from mutlistep_methods import AdamsMethods


def drange(start, stop, step):
    r = start
    while r < stop:
        n = floor(fabs(log10(0.02))) + 1
        # r = ceil(r * 10 ** n) / 10 ** n
        yield r
        r += step
        r = ceil(r * 10 ** n) / 10 ** n


def test():
    test_dy_f = lambda x, y, z: z
    test_dz_g = lambda x, y, z: (2 * x * y) / (x ** 2 + 1)
    func = np.asarray([1, 1.607999216, 2.263994646, 3.015985963, 3.911973624, 5]).flatten()
    res = AdamsMethods.adams_method(test_dy_f, test_dz_g, 0.2, 0, 1, 3, 1)
    print(res)
    print(np.fabs(np.subtract(res[:, 1:2].flatten(), func)))
    print()

    res = AdamsMethods.adams_boshfort_method(test_dy_f, test_dz_g, 0.2, 0, 1, 3, 1)
    print(res)
    print(np.fabs(np.subtract(res[:, 1:2].flatten(), func)))
