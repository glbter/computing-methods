import numpy as np
from simple_iteration import SimpleIteration
from gauss_method import Gauss
from running_method import RunningMethod


matrix1 = [[2, -7, 8, -4, 57],
           [0, -1, 4, -1, 24],
           [3, -4, 2, -1, 28],
           [-9, 1, -4, 6, 12]]

matrix2 = [[7, -5, 0, 0, 0, 38],
           [-6, 19, -9, 0, 0, 14],
           [0, 6, -18, 7, 0, -45],
           [0, 0, -2, -12, -8, -58],
           [0, 0, 0, -3, -10, -8]]

matrix3 = [[-24, -6, 4, 7, 130],
           [-8, 21, 4, -2, 139],
           [6, 6, 16, 0, -84],
           [-7, -7, 5, 24, -165]]

e = 0.01


def println(something):
    print(np.asarray(something), '\n')


if __name__ == '__main__':
    print('Метод Гаусса')
    println(Gauss.solve(matrix1))
    println(Gauss.find_determinant(matrix1))
    println(Gauss.find_inverse(matrix1))

    print('Метод прогону')
    println(RunningMethod.solve(matrix2))

    print('Метод простих ітерацій')
    println(SimpleIteration.solve(matrix3, e))
    print('Метод Зейделя')
    println(SimpleIteration.zeidel_solve(matrix3, e))
