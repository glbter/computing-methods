import numpy as np

test = [[1, 2, -3, 14],
        [2, -3, 1, -7],
        [-1, -1, 5, -18]]

test = [[1, 2, -3, 14],
        [2, -3, 1, -7],
        [-1, -1, 5, -18]]

test2 = [[10, 1, 1, 12],
         [2, 10, 1, 13],
         [2, 2, 10, 14]]

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


def matrix_copy(matrix):
    return [row.copy() for row in matrix]


def matrix_one(n):
    diag_m = []
    for i in range(n):
        row = []
        for j in range(n):
            if i != j:
                row.append(0)
            else:
                row.append(1)
        diag_m.append(row.copy())
    return diag_m


def gauss_first_step(matrix):
    matrix = matrix_copy(matrix)
    rows = len(matrix)
    columns = len(matrix[0])
    # first step
    # d for diagonal
    diag_m = matrix_one(rows)
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


def gauss_second_step(matrix):
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


def gauss_determinant(matrix):
    matrix, _ = gauss_first_step(matrix)
    product = 1
    for i in range(len(matrix)):
        product *= matrix[i][i]
    return product


def gauss_inverse(matrix):
    matrix = matrix_copy(matrix)
    n = len(matrix)
    origin, diag = gauss_first_step(matrix)
    x = []
    for i in range(n):
        for j in range(n):
            origin[j][-1] = diag[j][i]
        x.append(gauss_second_step(origin))
    arr = np.asarray(x).transpose().tolist()
    return arr


def gauss(matrix):
    matrix = matrix_copy(matrix)
    matrix, _ = gauss_first_step(matrix)
    return gauss_second_step(matrix)


if __name__ == '__main__':
    print(gauss(test))
    print(gauss_determinant(test2))
    print(gauss(test2))
    print(gauss_inverse(test2))
    # for i in range(4, 0, -1):
    #     print(i)
    #
    # for i,d in enumerate(range(0, 4)):
    #     print(i, d)

