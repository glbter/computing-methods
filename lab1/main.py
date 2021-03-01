import numpy as np
from math import log, e, log10, fabs, pow
import timeit
import matplotlib.pyplot as plt
plt.style.use('classic')
import seaborn as sns
sns.set()


def find_intervals(func, left=-100, right=100, step=1):
    return ((i, i+step)
            for i in range(left, right, step)
            if func(i) * func(i+step) < 0)


def time_count(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = timeit.default_timer()
            print(name)
            result = func(*args, **kwargs)
            print("Computing time is :", (timeit.default_timer() - start) * 1_000)
            return result
        return wrapper
    return decorator


def with_precision(num, prec):
    prec = int(10 ** fabs(log10(prec)))
    return int((num * prec) + 0.5) / prec


@time_count(name="Half division method")
def half_division(func, left, right, precision):
    while right - left > 2 * precision:
        f_left = func(left)
        f_right = func(right)
        avg = (left + right) / 2
        f_avg = func(avg)
        if f_left * f_avg < 0:
            right = avg
        elif f_avg * f_right < 0:
            left = avg
        else:
            break
    return with_precision((left + right) / 2, precision)


@time_count(name="Newton's method")
def newtons_method(func, derivative1, derivative2, x, precision):
    if func(x) * derivative2(x) <= 0:
        return "sequence is not convergent"
    x_prev = x
    x -= func(x) / derivative1(x)
    while fabs(x - x_prev) >= precision:
        x_prev = x
        x = x - func(x) / derivative1(x)
    return with_precision(x, precision)


@time_count(name="iteration method")
def iteration_method(x_func, x, q, precision):
    mul = q / (1-q)
    x_prev = x
    x = x_func(x)
    try:
        while fabs(x - x_prev) >= precision:
            x_prev = x
            x = x_func(x)
    except OverflowError:
        return "sequence is not convergent"
    return with_precision(x, precision)


def graph_interceptions():

    x = np.linspace(-1.999, 2, 5000)
    y = x**4
    z = [log(_ + 2) + 0.5 for _ in x]

    # plot the functions
    plt.plot(x, z, 'm', label='log(x + 2) + 0.5')
    plt.plot(x, y, 'c', label='x^4')

    plt.legend(loc='upper left')
    plt.show()


def plot_phi_x1():
    x = np.linspace(-1.3, 2, 5000)
    # phi = [pow(fabs(log(_ + 2) + 0.5), 0.25) for _ in x]
    f_ = lambda x: 1 / (2 * pow(2 * (2 * log(x+2) + 1)**3, 0.25) * (x + 2))
    phi = [f_(z) for z in x]
    # plot the functions
    plt.plot(x, phi, 'c', label='1 / (2 * (2(2log(x+2) + 1)^3)^0.25 * (x+2))')
    # plt.plot(x, phi, 'c', label='(log(x + 2) + 0.5)^0.25')
    plt.legend(loc='upper left')
    plt.show()


def plot_phi_x2():
    x = np.linspace(-1, -0.75, 5000)
    #phi = e ** (x**4 - 0.5) - 2
    phi = (4 * x**3) * (e ** (x**4 - 0.5))
    # plot the functions
    plt.plot(x, phi, 'c', label='4x^3 * e^(x^4-0.5)')
    # plt.plot(x, phi, 'c', label='e^(x^4 - 0.5) - 2')
    plt.legend(loc='upper left')
    plt.show()


def main():
    #graph_interceptions()
    plot_phi_x1()
    plot_phi_x2()

    def y(x):
        return log(x+2) - x**4 + 0.5

    def y_(x):
        return 1/(x+2) - (4 * x**3)

    def y__(x):
        return -1/((x+2)**2) - (12 * x**2)

    def xf1(x):
        return pow(log(x + 2) + 0.5, 0.25)

    def xf2(x):
        return e ** (x**4 - 0.5) - 2

    left1, right1 = -1, -0.75
    left2, right2 = 1, 1.35

    def print_results(left, right, pres, func, func_, func__, q_iter=0.5, xf=lambda x: x):
        print(f"Point from interval [{left}, {right}]")
        print(half_division(func, left, right, pres))
        print(newtons_method(func, func_, func__, left, pres))
        print(iteration_method(xf, (left1 + right1) / 2, q_iter, pres))
        print()

    print("f(x) = log(x + 2) - x ** 4 + 0.5")
    print("f'(x) = 1 / (x + 2) - (4 * x ** 3)")
    print()
    print("graphical way")
    print("phi(x) = (log(x + 2) + 0.5) ^ 0.25")
    print_results(left1, right1, 0.001, y, y_, y__, xf=xf1)
    print("phi(x) = (log(x + 2) + 0.5) ^ 0.25")
    print_results(left2, right2, 0.001, y, y_, y__, xf=xf1)
    print("phi(x) = e^(x^4 - 0.5) - 2")
    print_results(left1, right1, 0.001, y, y_, y__, xf=xf2)
    print("phi(x) = e^(x^4 - 0.5) - 2")
    print_results(left2, right2, 0.001, y, y_, y__, xf=xf2)
    print()
    print("analytic way")
    for a,b in find_intervals(y, -1):
        print_results(a, b, 0.001, y, y_, y__, xf=xf1)

    # pres = 0.001
    # print(iteration_method(xf1, (left1 + right1) / 2, 0.9, pres))
    # print(iteration_method(xf1, (left2 + right2) / 2, 0.9, pres))


if __name__ == "__main__":
    main()

    #
    # def y_test(x):
    #     return e ** (2 * x) + 3 * x - 4
    #
    # def y_test_(x):
    #     return 2 * e ** (2 * x) + 3
    #
    # def y_test__(x):
    #     return 4 * e ** (2 * x)
    #
    # def xf_test(x):
    #     return log(4 - 3 * x) / 2
    # print(half_division(y_test, 0.4, 0.6, 0.001))
    # print(newtons_method(y_test, y_test_, y_test__, 0.6, 0.001))
    # print(iteration_method(xf_test, 0.475, 0.64, 0.001))

