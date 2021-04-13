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