from functools import reduce
import numpy as np

def compute_jacobian(j, x1, x2):
    return [[elem(x1, x2) for elem in row] for row in j]


def compute_system(s, x1, x2):
    return [func(x1, x2) for func in s]


def check_stop(x, x_prev) -> int:
    return reduce(max, (np.fabs(xk1-xk) for xk1, xk in zip(x, x_prev)))


def compute_special_jacobian(j, x1, x2):
    computed_j = ((np.fabs(func(x1, x2)) for func in row) for row in j)
    sums = (sum(row) for row in computed_j)
    return reduce(max, sums)
