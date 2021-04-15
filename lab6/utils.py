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


def matrix_copy(matrix):
    return [row.copy() for row in matrix]
