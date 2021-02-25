import pandas as pd
import numpy as np

y = lambda x: 1 / x
d1 = [0.1, 0.5, 0.9, 1.3]
d2 = [0.1, 0.5, 1.1, 1.3]
x_ = 0.8
y = [y(x) for x in d1]

dt1 = pd.DataFrame([d1, y], index=['x', 'y'], columns=['1', '2', '3', '4'])
dt2 = pd.DataFrame([d2, y], index=['x', 'y'], columns=['1', '2', '3', '4'])


def func_mul(f1, f2):
    return lambda x: f1(x) * f2(x)


def func_sum(f1, f2):
    return lambda x: f1(x) + f2(x)


def func_const_mul(f, const):
    return lambda x: f(x) * const


def vyraz(arr, j):
    def f(c):
        def inner(x):
            return x-c
        return inner

    res = lambda x: 1
    for i in range(len(arr)):
        if i != j:
            res = func_mul(res, f(arr[i]))

    #result = [f(arr[i]) for i in range(len(arr)) if i != j][0]
    #[(lambda x: x-arr[i]) for i in range(len(arr)) if i != j][0]
    return res


def sche_vyraz(df: pd.DataFrame):
    x = df.to_numpy()[0]
    y = df.to_numpy()[1]
    res = lambda z: 0
    for i in range(len(x)):
        f = vyraz(x, i)
        y_ = y[i]
        x_ = (f(x[i]))
        const = y_ / x_
        res = func_sum(res, func_const_mul(f, const))
    return res

def main():
    print(dt1)
    print(vyraz(d1, 0)(0.5))
    print(sche_vyraz(dt1)(0.3))



if __name__ == "__main__":
    main()