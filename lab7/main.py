import matplotlib.pyplot as plt
from test_data import *
from newton_method import *
from iteration_method import *


def draw(system):
    plt.figure()  # Create a new figure window
    xlist = np.linspace(-2.0, 2.0, 100)  # Create 1-D arrays for x,y dimensions
    ylist = np.linspace(-2.0, 2.0, 100)
    X, Y = np.meshgrid(xlist, ylist)  # Create 2-D grid xlist,ylist values
    F = system[0](X, Y)  # 'Circle Equation
    plt.contour(X, Y, F, [0], colors='r', linestyles='solid')
    F = system[1](X, Y)
    plt.contour(X, Y, F, [0], colors='k', linestyles='solid')
    plt.plot(0.70224654, 0.84613601, 'bx')
    plt.plot(-0.42734707, -0.45541399, 'bx')
    plt.show()


def pp(system, jacobian, x1, x2):
    print(newtons_method3(system, jacobian, x1, x2, 0.0001))
    print(newtons_method(system, jacobian, x1, x2, 0.0001))
    print(newton_method(system, jacobian, x1, x2, 0.0001))


def print_res(system, jacobian, x1, x2, prec):
    res = newton_method(system, jacobian, x1, x2, prec)
    print("result")
    print(res)
    print("precision")
    for f in system:
        print(f(res[0], res[1]))
    print()


if __name__ == "__main__":
    print_res(system, jacobian, -0.5, -0.5, 0.0001)
    print_res(system, jacobian, 1, 1, 0.0001)
    # draw(system)
