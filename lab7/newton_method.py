from utils import *


# methods using f(xk) J(xk)*dx = 0 solving
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
    return x


def newtons_method3(system, j, x1, x2, precision):
    x = np.asarray([x1, x2])
    while True:
        x_prev = x
        ss = compute_system(system, x[0], x[1])
        ss = [-x for x in ss]
        js = compute_jacobian(j, x[0], x[1])
        dx = np.linalg.solve(js, ss)
        x = np.asarray([x + dx for x, dx in zip(x, dx)])
        if check_stop(x, x_prev) < precision:
            break
    return x#[np.round(_, int(-log10(precision))) for _ in x]


# method using det J and A1 A2 for solving
def newton2(system, j, x1, x2, prec):
    x = np.asarray([x1, x2])
    system = np.asarray(system)
    j = np.asarray(j)
    a1, a2 = j[:, 1:], j[:, :1]
    a1 = np.asarray([system.flatten(), a1.flatten()]).T
    a2 = np.asarray([a2.flatten(), system.flatten()]).T
    while True:
        matrixes = []
        for matrix in (j, a1, a2):
            computed = compute_jacobian(matrix, x[0], x[1])
            matrixes.append(np.linalg.det(computed))
        delta = np.asarray([matrixes[1]/matrixes[0], matrixes[2]/matrixes[0]])
        x_prev = x
        x = x - delta
        if check_stop(x, x_prev) < prec:
            break
    return x
