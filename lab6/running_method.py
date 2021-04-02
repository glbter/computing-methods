from gauss_method import matrix_copy


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

class RunningMethod:
    @staticmethod
    def solve(matrix):
        matrix = matrix_copy(matrix)
        n = len(matrix)
        pq = []
        p, q = 0, 0
        P, Q = 'p', 'q'
        for i in range(0, n):
            a = matrix[i][i-1] if (i != 0) else 0
            b = matrix[i][i]
            c = matrix[i][i+1] if (i != n-1) else 0
            d = matrix[i][-1]
            q = (d - a*q) / (b + a*p)
            p = -c / (b + a*p)
            pq.append({P:p, Q:q})

        X, x = [], 0
        for i in range(n-1, 0-1, -1):
            q = pq[i][Q]
            p = pq[i][P]
            x = p*x + q
            X.append(x)

        X.reverse()
        return X