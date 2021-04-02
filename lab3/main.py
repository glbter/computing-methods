import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
import seaborn as sns
sns.set()
from lab6.main import matrix_copy
from lab2.main import func_sum, func_const_mul, diff_func


def diff_func_invert(xi):
    def inner(x):
        return xi - x
    return inner


def cube_func(xi):
    def inner(x):
        return (x - xi)**3
    return inner


def cube_func_invert(xi):
    def inner(x):
        return (xi - x)**3
    return inner


def spline_interpolation(table):
    table = matrix_copy(table)
    x, y = table[0], table[1]
    n = len(x)
    slae = np.zeros((n-2, n-1))
    for i in range(1, n-1): # [1, n-1]
        hi = x[i] - x[i-1]
        hi_1 = x[i+1] - x[i]
        dy = y[i] - y[i-1]
        dy_1 = y[i + 1] - y[i]

        c = dy_1/hi_1 - dy/hi
        slae[i-1][n-2] = c

        qi = (hi + hi_1) / 3
        slae[i-1][i-1] = qi

        if i != 1:
            qi_minus_1 = hi / 6
            slae[i-1][i-2] = qi_minus_1
        if i != n-2:
            qi_plus_1 = hi_1 / 6
            slae[i-1][i] = qi_plus_1

    q = np.linalg.solve(slae[:n, :n-2], slae[:n, n-2:])
    q = np.reshape(q, n-2)
    q = np.insert(q, 0, 0)
    q = np.insert(q, n-1, 0)
    return np.append(table, [q], axis=0)


def spline_function(table):
    x = table[0]
    y = table[1]
    q = table[2]
    n = len(x)
    funcs = []
    for i in range(1, n):
        qi = q[i]
        qi_prev = q[i-1]
        hi = x[i] - x[i-1]
        c1 = qi_prev / (6*hi)
        c2 = qi / (6*hi)
        c3 = y[i-1]/hi - qi_prev*hi/6
        c4 = y[i]/hi - qi*hi/6

        p1 = func_const_mul(cube_func_invert(x[i]), c1)
        p2 = func_const_mul(cube_func(x[i-1]), c2)
        p3 = func_const_mul(diff_func_invert(x[i]), c3)
        p4 = func_const_mul(diff_func(x[i-1]), c4)

        sx = func_sum(func_sum(p1, p2), func_sum(p3, p4))
        funcs.append(sx)

    ranges = []
    for i in range(1, len(x)):
        ranges.append((x[i-1], x[i]))
    sx = (tuple(ranges), tuple(funcs))
    return sx


def plot_spline_function(spline_table):
    table = spline_table
    for r, f in zip(table[0], table[1]):
        x = np.linspace(r[0], r[1], 1000)
        y = f(x)
        plt.plot(x, y, 'c')
    plt.show()


def count_spline(spline_table, x):
    table = spline_table
    ranges = table[0]
    f = table[1]
    for i in range(len(ranges)):
        r0 = ranges[i][0]
        r1 = ranges[i][1]
        res = f[i](x)
        if (x > r0) and (x <= r1): return res
        if (i == 0) and (x <= r1): return res
        if (i == (len(ranges)-1)) and (x > r0): return res


if __name__ == '__main__':
    test = [[1, 2, 3, 4, 5],
            [1, 3, 6, 9, 21]]

    x = 0.8
    table = [[0.1, 0.5, 0.9, 1.3, 1.7],
             [10, 2, 1.11111, 0.76923, 0.58824]]

    res = spline_interpolation(table)
    print("spline table:")
    print(res)
    res = spline_function(res)
    plot_spline_function(res)
    print()
    print(f"x = {x},   S(x) = {count_spline(res, x)}")



