import numpy as np
from functools import reduce
from math import fabs, sqrt


class SimpleIteration:
    @staticmethod
    def _check_convergence(matrix) -> bool:
        si = SimpleIteration
        return si._matrix_norm_1(matrix) < 1 \
               or si._matrix_norm_2(matrix) < 1 \
               or si._matrix_norm_c(matrix) < 1

    @staticmethod
    def _check_error(vector, err) -> bool:
        si = SimpleIteration
        return si._vector_norm_1(vector) <= err \
               or si._vector_norm_2(vector) <= err \
               or si._vector_norm_c(vector) <= err

    @staticmethod
    def _vector_norm_1(vector) -> int:
        return sum((fabs(x) for x in vector))

    @staticmethod
    def _vector_norm_2(vector) -> int:
        return sqrt(sum((x**2 for x in vector)))

    @staticmethod
    def _vector_norm_c(vector) -> int:
        return reduce(max, (fabs(x) for x in vector))

    @staticmethod
    def _matrix_norm_1(matrix) -> int:
        return SimpleIteration._matrix_norm_c(np.array(matrix).transpose())

    @staticmethod
    def _matrix_norm_2(matrix) -> int:
        ar = np.asarray(matrix).flatten()
        return SimpleIteration._vector_norm_2(ar)

    @staticmethod
    def _matrix_norm_c(matrix) -> int:
        return reduce(max, (sum((fabs(x) for x in vect)) for vect in matrix))

    @staticmethod
    def _equivalent(matrix):
        B = []
        n = len(matrix)
        for i in range(n):
            a = matrix[i][i]
            b = matrix[i][-1]
            beta = b / a
            B.append(beta)

        A = []
        for i in range(n):
            a_ii = matrix[i][i]
            A_row = []
            for j in range(n):
                if i != j:
                    a_ij = matrix[i][j]
                    alpha = -a_ij / a_ii
                else:
                    alpha = 0
                A_row.append(alpha)
            A.append(A_row.copy())
        return A, B

    @staticmethod
    def solve(matrix, precision):
        si = SimpleIteration
        alpha, beta = si._equivalent(matrix)

        if not si._check_convergence(alpha):
            raise RuntimeWarning

        alpha = np.asarray(alpha)
        beta = np.asarray(beta)

        x = beta
        while True:
            x_prev = x
            x = beta + alpha.dot(x)
            if si._check_error(x - x_prev, precision):
                break
        return x

    @staticmethod
    def zeidel_solve(matrix, precision):
        si = SimpleIteration
        alpha, beta = si._equivalent(matrix)

        if not si._check_convergence(alpha):
            raise RuntimeWarning

        alpha = np.asarray(alpha)
        beta = np.asarray(beta)
        e = np.identity(len(matrix))
        c = np.triu(alpha)
        b = np.tril(alpha, -1)

        x = beta
        while True:
            x_prev = x
            inv = np.linalg.inv(e-b)
            x = inv.dot(c).dot(x) + inv.dot(beta)
            if si._check_error(x - x_prev, precision):
                break
        return x
