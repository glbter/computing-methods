from math import fabs
from test_data import *
from mutlistep_methods import *


def compute_error(func, table_func: np.array) -> np.array:
    return np.asarray([fabs(func(x)-y) for x, y in table_func])


def print_res(name: str, res):
    print(name)
    print(res)
    print(compute_error(function, res))
    print()


def compute(f, g, h: int, x0: int, y: int, z: int, xk: int):
    print_res("Runge-Kutt method", RungeKutt.runge_kutt(f, g, h, x0, y, z, xk))
    print_res("Adams method", AdamsMethods.adams_method(f, g, h, x0, y, z, xk))
    print_res("Adams-Boshfort-Moulton method", AdamsMethods.adams_boshfort_method(f, g, h, x0, y, z, xk))


if __name__ == '__main__':
    compute(dy_f, dz_g, 0.2, 1, 3, 2, 2)


