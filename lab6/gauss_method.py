import numpy as np
from utils import matrix_copy


class Gauss:
    @staticmethod
    def _first_step(matrix):
        matrix = matrix_copy(matrix)
        rows = len(matrix)
        columns = len(matrix[0])
        # first step
        # d for diagonal
        diag_m = np.identity(rows)
        for i, d in enumerate(range(0, rows-1)):
            if (row := matrix[i])[i] == 0:
                for j in range(i, rows):
                    if (tmp := row[j]) != 0:
                        row[j] = row[i]
                        row[i] = tmp
                else:
                    raise AttributeError
            for j in range(i+1, rows):
                mul = -1
                if (elem := matrix[j][d]) != 0:
                    mul *= elem / matrix[i][d] # divide by main
                    for k in range(0, columns):
                        matrix[j][k] += mul * matrix[i][k]
                        if k < rows:
                            diag_m[j][k] += mul * diag_m[i][k]
        return matrix, diag_m

    @staticmethod
    def _second_step(matrix):
        matrix = matrix_copy(matrix)
        rows = len(matrix)
        columns = len(matrix[0])
        # second step
        x = []
        for i in range(rows - 1, -1, -1):
            const = matrix[i][columns - 1]
            for n, k in enumerate(range(columns - 2, i, -1)):
                const -= matrix[i][k] * x[n]
                matrix[i][columns - 1] = const
            xi = const / matrix[i][i]
            x.append(xi)
        x.reverse()
        return x

    @staticmethod
    def find_determinant(matrix):
        matrix, _ = Gauss._first_step(matrix)
        product = 1
        for i in range(len(matrix)):
            product *= matrix[i][i]
        return product

    @staticmethod
    def find_inverse(matrix):
        matrix = matrix_copy(matrix)
        n = len(matrix)
        origin, diag = Gauss._first_step(matrix)
        x = []
        for i in range(n):
            for j in range(n):
                origin[j][-1] = diag[j][i]
            x.append(Gauss._second_step(origin))
        arr = np.asarray(x).transpose().tolist()
        return arr

    @staticmethod
    def solve(matrix):
        matrix = matrix_copy(matrix)
        matrix, _ = Gauss._first_step(matrix)
        return Gauss._second_step(matrix)
