import numpy as np
from simple_iteration import SimpleIteration
from gauss_method import Gauss
from running_method import RunningMethod
from test_data import *


def println(something):
    print(np.asarray(something), '\n')


if __name__ == '__main__':
    print('Метод Гаусса')
    println(Gauss.solve(matrix1))
    print("Детермінант матриці")
    println(Gauss.find_determinant(matrix1))
    print("Обернена матриця")
    println(Gauss.find_inverse(matrix1))

    print('Метод прогону')
    println(RunningMethod.solve(matrix2))

    print('Метод простих ітерацій')
    println(SimpleIteration.solve(matrix3, e))
    print('Метод Зейделя')
    println(SimpleIteration.zeidel_solve(matrix3, e))
