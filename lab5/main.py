import numpy as np
from math import fabs
# x**2 * y'' + x*y' - y  -3x**2 = 0
# y(1) =  3
# y'(1) = 2
# x = [1, 2], h = 0.1
# y = x**2 + x + 1/x
# y'' = (3x^2 + y - xy') / x^2
# z' = (3x^2 + y - xz) / x^2
# y' = z

# test_dy_f = lambda x, y, z: z
# test_dz_g = lambda x, y, z: (2*x*y) / (x**2 + 1)


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


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


def runge_kutt(f, g, h: int, x0: int, y: int, z: int, xk: int) -> np.array:
    xy = [(x0, y)]
    for x in np.arange(x0, xk, h):
        k1 = h * f(x, y, z)
        l1 = h * g(x, y, z)
        k2 = h * f(x + h / 2, y + k1 / 2, z + l1 / 2)
        l2 = h * g(x + h / 2, y + k1 / 2, z + l1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2, z + l2 / 2)
        l3 = h * g(x + h / 2, y + k2 / 2, z + l2 / 2)
        k4 = h * f(x + h, y + k3, z + l3)
        l4 = h * g(x + h, y + k3, z + l3)

        dy = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        dz = 1/6 * (l1 + 2*l2 + 2*l3 + l4)
        y += dy
        z += dz
        xy.append((x+h, y))
    return np.asarray(xy)


def adams_method(f, g, h: int, x0: int, y: int, z: int, xk: int) -> np.array:
    xyzgf = [(x0, y, z, y, z)]
    for x in drange(x0, x0 + 3*h, h):
        k1 = h * f(x, y, z)
        l1 = h * g(x, y, z)
        k2 = h * f(x + h / 2, y + k1 / 2, z + l1 / 2)
        l2 = h * g(x + h / 2, y + k1 / 2, z + l1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2, z + l2 / 2)
        l3 = h * g(x + h / 2, y + k2 / 2, z + l2 / 2)
        k4 = h * f(x + h, y + k3, z + l3)
        l4 = h * g(x + h, y + k3, z + l3)

        dy = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        dz = 1/6 * (l1 + 2*l2 + 2*l3 + l4)
        y += dy
        z += dz

        xyzgf.append((x+h, y, z, g(x,y,z), f(x,y,z)))

    for x in drange(x0 + 4*h, xk, h):
        fk = xyzgf
        yk = xyzgf[-1][1] + h/24 * (55*fk[-1][-1] - 59*fk[-2][-1] + 37*fk[-3][-1] - 9*fk[-4][-1])
        gk = xyzgf
        zk = xyzgf[-1][2] + h/24 * (55*gk[-1][-2] - 59*gk[-2][-2] + 37*gk[-3][-2] - 9*gk[-4][-2])
        # yk = xyzgf[-1][1] + h/24 * (55*gk[-1][-2] - 59*gk[-2][-2] + 37*gk[-3][-2] - 9*gk[-4][-2])
        # zk = xyzgf[-1][2] + h/24 * (55*fk[-1][-1] - 59*fk[-2][-1] + 37*fk[-3][-1] - 9*fk[-4][-1])
        fk1 = f(x, yk, zk)
        gk1 = g(x, yk, zk)
        xyzgf.append((x+h, yk, zk, gk1, fk1))
    return np.asarray(xyzgf)


def compute_error(func, table_func: np.array) -> np.array:
    return np.asarray([fabs(func(x)-y) for x, y in table_func])


if __name__ == '__main__':
    # res = runge_kutt(test_dy_f, test_dz_g, 0.2, 0, 1, 3, 1)
    res = runge_kutt(dy_f, dz_g, 0.2, 1, 3, 2, 2)
    print(res)
    print(compute_error(function, res))
    # res = adams_method(test_dy_f, test_dz_g, 0.2, 0, 1, 3, 1)
    res = adams_method(dy_f, dz_g, 0.2, 1, 3, 2, 2)
    print(res)
    print(compute_error(function, res[:, :2]))
    # print(adams_method(test_dy_f, test_dz_g, 0.2, 0, 1, 3, 1))
