import numpy as np
from math import tan, cos, fabs, sin, log10
from lab6.simple_iteration import SimpleIteration
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns

system = [
    lambda x1, x2: 2 * (x1**2) - x1 + x2**2 - 1,
    lambda x1, x2: x2 - tan(x1)
]

jacobian = [
    [lambda x1, x2: 4*x1 - 1, lambda x1, x2: 2*x2],
    [lambda x1, x2: -1/(cos(x1)**2), lambda x1, x2: 1]
]

test_system = [
    lambda x1, x2: 0.1 * x1**2 + x1 + 0.2 * x2**2 - 0.3,
    lambda x1, x2: 0.1 * x1**2 + x2 - 0.1 * x1 * x2 - 0.7
]

def f1dx1_test(x1, x2):
    return 0.2 * x1 + 1

test_jacobian = [
    [f1dx1_test, lambda x1, x2: 0.4 * x2],
    [lambda x1, x2: 0.2 * x1 - 0.1 * x2, lambda x1, x2: 1 - 0.1 * x1]
]

test_special = [
    lambda x1, x2: 0.3 - 0.1*x1**2 - 0.2*x2**2,
    lambda x1, x2: 0.7 - 0.2*x1**2 + 0.1*x1*x2
]


def compute_jacobian(j, x1, x2):
    return [[elem(x1, x2) for elem in row] for row in j]


def compute_system(s, x1, x2):
    return [func(x1, x2) for func in s]


def check_stop(x, x_prev) -> int:
    return reduce(max, (fabs(xk1-xk) for xk1, xk in zip(x, x_prev)))


def newtons_method(system, j, x1, x2, precision):
    x = np.asarray([x1, x2])
    while True:
        s = compute_system(system, x[0], x[1])
        inv_system = [-func for func in s]
        delta_x = np.linalg.solve(compute_jacobian(j, x[0], x[1]), inv_system)
        x_prev = x
        x = x + delta_x
        if check_stop(x, x_prev) < precision:
            break
    return [np.round(_, int(-log10(precision))) for _ in x]


def newtons_method3(system, j, x1, x2, precision):
    x = np.asarray([x1, x2])
    while True:
        x_prev = x
        ss = compute_system(system, x[0], x[1])
        ss = [-x for x in ss]
        js = compute_jacobian(j, x[0], x[1])
        dx = np.linalg.solve(js, ss)
        x = x + dx
        if check_stop(x, x_prev) < precision:
            break
    return [np.round(_, int(-log10(precision))) for _ in x]


def iteration_method(phi, x1, x2, q, precision):
    x1 = phi[0](x1, x2)
    x2 = phi[1](x1, x2)
    x = np.asarray([x1, x2])
    while True:
        x_prev = x
        x1 = phi[0](x_prev[0], x_prev[1])
        x2 = phi[1](x_prev[0], x_prev[1])
        x = np.asarray([x1, x2])
        if q/(1-q) * check_stop(x, x_prev) < precision:
            break
    return x


def draw(func1, func2):
    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)
    # y = np.linspace(-1, 1, 100)
    plt.plot(x1, func1(x1, x2))
    plt.plot(x2, func1(x1, x2))
    plt.show()


if __name__ == "__main__":
    # [0.19641325 0.70615447]
    test2_system = [
        lambda x, y: sin(2*x - y) - 1.2*x - 0.4,
        lambda x, y: 0.8* x**2 + 1.5* y**2 - 1
    ]
    test2_jacobian = [
        [lambda x, y: 2*cos(2*x - y) - 1.2, lambda x, y: cos(2*x - y)],
        [lambda x, y: 1.6*x, lambda x, y: 3*y]
    ]
    # [0.19534837 0.7100547]
    # print(newtons_method(test_system, test_jacobian, 0.25, 0.75, 0.0001))
    # print(newtons_method3(test_system, test_jacobian, 0.25, 0.75, 0.0001))
    # print(newtons_method3(test2_system, test2_jacobian, 0.4, -0.75, 0.0001))
    # print(newtons_method(test2_system, test2_jacobian, 0.4, -0.75, 0.0001))
    # test2_result = (newtons_method(test2_system, test2_jacobian, 0.4, -0.75, 0.0001))
    test_result = newtons_method3(test_system, test_jacobian, 0.25, 0.75, 0.0001)
    print(test_result)
    print(test_system[0](test_result[0], test_result[1]))
    print(test_system[1](test_result[0], test_result[1]))
    test_result = iteration_method(test_special, 0.25, 0.75, 0.5, 0.0001)
    print(test_result)
    print(test_system[0](test_result[0], test_result[1]))
    print(test_system[1](test_result[0], test_result[1]))
    # print(iteration_method(test_special, 0.25, 0.75, 0.5, 0.0001))
    # print(newtons_method(system, jacobian))
    # draw(system[0], system[1])
    pass