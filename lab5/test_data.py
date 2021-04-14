import numpy as np
from mutlistep_methods import AdamsMethods


def dy_f(x, y, z):
    return z


def dz_g(x, y, z):
    return (3*x**2 + y - x*z) / x**2


def function(x):
    return x**2 + x + 1/x


def test_dy_f(x, y, z):
    return z


def test_dz_g(x, y, z):
    return (2*x*z) / (x**2 + 1)


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