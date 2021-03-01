import pandas as pd
from math import fabs


def func_mul(f1, f2):
    return lambda x: f1(x) * f2(x)


def func_sum(f1, f2):
    return lambda x: f1(x) + f2(x)


def func_const_mul(f, const):
    return lambda x: f(x) * const


def diff_func(c):
    def inner(x):
        return x-c
    return inner


def diff_polynomial(arr, j):
    res = lambda x: 1
    for i in range(len(arr)):
        if i != j:
            res = func_mul(res, diff_func(arr[i]))
    # python bug : lambda doesn't work in cycle as should be
    #result = [f(arr[i]) for i in range(len(arr)) if i != j][0]
    #[(lambda x: x-arr[i]) for i in range(len(arr)) if i != j][0]
    return res


def lagrange_polynomial(df: pd.DataFrame):
    x = df.to_numpy()[0]
    y = df.to_numpy()[1]
    res = lambda z: 0
    for i in range(len(x)):
        f = diff_polynomial(x, i)
        y_ = y[i]
        x_ = (f(x[i]))
        const = y_ / x_
        res = func_sum(res, func_const_mul(f, const))
    return res


def newton_table(n):
    def local_table(n):
        table = []
        for i in range(n):
            list = []
            for j in range(i+1):
                list.append(0)
            table.append(list)
            if i == n-1:
                table.append(list.copy())
        table.reverse()
        return table
    return [local_table(n), local_table(n)]


def fill_table(table, x, y):
    for i in range(len(x)):
        table[0][0][i] = x[i]
    for i in range(len(y)):
        table[0][1][i] = y[i]
    for i in range(len(x)):
        table[1][0][i] = True
        table[1][1][i] = True
    return table


def divided_difference(i1, i2, table):
    var_table = table[0]
    bool_table = table[1]
    x = var_table[0]
    x1, x2 = x[i1], x[i2]
    dd = divided_difference
    power = i2 - i1 # range +1 beetween i1+1, i2 or i1, i2-1
    f2 = var_table[power][i1]
    f1 = var_table[power][i1 + 1]
    passed_f2 = bool_table[power][i1]
    passed_f1 = bool_table[power][i1 + 1]
    if not passed_f1:
        f1 = dd(i1 + 1, i2, table)
        var_table[power][i1 + 1] = f1
        bool_table[power][i1 + 1] = True
    if not passed_f2:
        f2 = dd(i1, i2 - 1, table)
        var_table[power][i1] = f2
        bool_table[power][i1] = True
    return (f1 - f2) / (x2 - x1)


def newtons_polynomial(df: pd.DataFrame):
    x = df.to_numpy()[0]
    y = df.to_numpy()[1]
    lx = len(x)
    if lx != len(y): raise AttributeError
    table = newton_table(lx)
    table = fill_table(table, x, y)
    res = divided_difference(0, lx - 1, table)
    table[0][lx][0] = res
    table[1][lx][0] = True
    vtable = table[0]

    f = lambda x: 0
    for i in range(lx):
        mul = vtable[i+1][0]
        func = lambda x: 1
        if i > 0:
            for j in range(i):
                func = func_mul(func, diff_func(vtable[0][j]))
        func = func_const_mul(func, mul)
        f = func_sum(f, func)
    return f



def main():
    func = lambda x: 1 / x
    d1 = [0.1, 0.5, 0.9, 1.3]
    d2 = [0.1, 0.5, 1.1, 1.3]

    y1 = [func(x) for x in d1]
    y2 = [func(x) for x in d2]
    dt1 = pd.DataFrame([d1, y1], index=['x', 'y'], columns=['1', '2', '3', '4'])
    dt2 = pd.DataFrame([d2, y2], index=['x', 'y'], columns=['1', '2', '3', '4'])
    x_ = 0.8
    L1 = lagrange_polynomial(dt1)
    yL_1 = L1(x_)
    err1 = fabs(func(x_) - yL_1)
    print(yL_1, err1)

    N = newtons_polynomial(dt2)
    yN_1 = N(x_)
    err2 = fabs(func(x_) - yN_1)
    print(yN_1, err2)
    #
    # L2 = lagrange_polynomial(dt2)
    # y_2 = L2(x_)
    # err2 = fabs(func(x_) - y_2)
    # print(y_2, err2)
    #print(newton_table(5))


if __name__ == "__main__":
    main()