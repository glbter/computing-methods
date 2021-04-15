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


if __name__ == "__main__":
    # [0.19641325 0.70615447]
    # [0.19534837 0.7100547]
    # print(newtons_method(test_system, test_jacobian, 0.25, 0.75, 0.0001))
    # print(newtons_method3(test_system, test_jacobian, 0.25, 0.75, 0.0001))
    # print(newtons_method3(test2_system, test2_jacobian, 0.4, -0.75, 0.0001))
    # print(iteration_method(test_special, 0.25, 0.75, 0.5, 0.0001))
    print()
    # print(compute_special_jacobian(special_jacobian, initial[0], initial[1]))
    # print(compute_special_jacobian(test_special_jacobian, 0.25, 0.75))
    # print(iteration_method(system_special, initial[0], initial[1], 0.5, 0.0001))
    # print(newtons_method3(system_special, jacobian, initial[0], initial[1], 0.0001))
    # print(newtons_method3(system_special, jacobian, -0.5, -0.5, 0.0001))
    # print(system_special[0](0.0001, -1))
    # print(system_special[1](0.0001, -1))
    #
    # print(compute_system(test_system, 0.19641401, 0.70615447))
    # print(compute_system(test_system, 0.19534837, 0.7100547))
    # print(compute_system(test_system, 0.19533946, 0.71003593))

    # [0.19533946 0.71003593]
    # print(newton2(test_system, test_jacobian, 0.25, 0.75, 0.0001))
    def pp(system, jacobian, x1, x2):
        print(newtons_method3(system, jacobian, x1, x2, 0.0001))
        print(newtons_method(system, jacobian, x1, x2, 0.0001))
        print(newton2(system, jacobian, x1, x2, 0.0001))
    pp(system, jacobian, -0.5, -0.5)
    # print(iteration_method(system_super_special2, -0.5, -0.5, 0.5, 0.0001))
    print()
    pp(system, jacobian, 1, 1)
    # print(iteration_method(system_super_special2, 1, 1, 0.5, 0.0001))
    print()
    pp(system, jacobian, 0, 0)
    # print(iteration_method(system_super_special, 0, 0, 0.5, 0.0001))
    print()
    pp(system, jacobian, 0.5, 0.5)
    # print(iteration_method(system_super_special, 0.5, 0.5, 0.5, 0.0001))
    # [-0.42734707 - 0.45541399]
    print(system[0](0.70224654, 0.84613601))
    print(system[1](0.70224654, 0.84613601))
    print(system[0](-0.42734707, -0.45541399))
    print(system[1](-0.42734707, -0.45541399))
    draw(system)
    pass