import matplotlib.pyplot as plt
from test_data import *
from newton_method import *
from iteration_method import *


def draw(system):
    plt.figure()  # Create a new figure window
    xlist = np.linspace(-0.75, 1.25, 200)  # Create 1-D arrays for x,y dimensions
    ylist = np.linspace(-1.5, 1.5, 300)
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


def print_newtons(system, jacobian, x1, x2, prec):
    print(f"Newton's method x1 = {x1}, x2 = {x2}")
    res = newton_method(system, jacobian, x1, x2, prec)
    cout(res)


def print_iteration(phi, x1, x2, q, pres):
    print(f"Iterations method x1 = {x1}, x2 = {x2}, q = {q}")
    res = iteration_method(phi, x1, x2, q, pres)
    cout(res)


def cout(res):
    print("result")
    print(res)
    print("precision")
    for f in system:
        print(f(res[0], res[1]))
    print()


if __name__ == "__main__":
    print_newtons(system, jacobian, -0.5, -0.5, 0.0001)
    print_newtons(system, jacobian, 0.6, 0.9, 0.0001)
    print_iteration(system_super_special, 0.6, 0.9, 0.8, 0.0001)

    # draw(system)
