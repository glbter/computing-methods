from math import sqrt, fabs


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def runge_method(i1, i2, p):
    return i2 + (i2 - i1)/(2**p - 1)


def left_rect_method(x0, xn, h, f):
    return sum([f(xi) for xi in drange(x0, xn, h)]) * h


def right_rect_method(x0, xn, h, f):
    return left_rect_method(x0+h, xn+h, h, f)


def avg_rect_method(x0, xn, h, f):
    first = x0 + h/2
    return left_rect_method(first, xn, h, f)


def trapeze_method(x0, xn, h, f):
    return h/2 * (f(x0) + f(xn) + 2*sum([f(xi) for xi in drange(x0+h, xn, h)]))


def simpsons_method(x0, xn, h, f):
    odd = sum([f(x) for x in drange(x0+h, xn, 2*h)])
    even = sum([f(x) for x in drange(x0+2*h, xn, 2*h)])
    return h/3 * (f(x0) + f(xn) + 4*odd + 2*even)


def results(x0, xn, h, f, exact):
    print(f'x0 = {x0}, xn = {xn}, h = {h}')
    lr = left_rect_method(x0, xn, h, f)
    rr = right_rect_method(x0, xn, h, f)
    ar = avg_rect_method(x0, xn, h, f)
    t = trapeze_method(x0, xn, h, f)
    s = simpsons_method(x0, xn, h, f)
    ss = lambda x: f" absolute error: {fabs(x-exact):.6f}"
    print("left rectangle method result: ", lr, ss(lr))
    print("right rectangle method result: ", rr, ss(rr))
    print("average rectangle method result: ", ar, ss(ar))

    print("trapeze method result: ", t, ss(t))
    print("simpsons method result: ", s, ss(s))
    print('---------------------------------')
    return [lr, rr, ar, t, s]


def print_runge(i1, i2, p, exact):
    i = [runge_method(i1, i2, p) for i1, i2, p in zip(i1, i2, p)]
    ss = lambda x: f" absolute error: {fabs(x - exact):.6f}"
    print("Runge method")
    print("left rectangle method result: ", i[0], ss(i[0]))
    print("right rectangle method result: ", i[1], ss(i[1]))
    print("average rectangle method result: ", i[2], ss(i[2]))
    print("trapeze method result: ", i[3], ss(i[3]))
    print("simpsons method result: ", i[4], ss(i[4]))
    print('---------------------------------')


if __name__ == '__main__':
    exact = 0.41797
    y = lambda x: 1 / sqrt((2 * x + 7) * (3 * x + 4))
    x0, xk = 0, 4
    h1, h2 = 0.1, 0.05

    i1 = results(x0, xk, h1, y, exact)
    i2 = results(x0, xk, h2, y, exact)
    p = [1, 1, 1, 2, 4]
    print_runge(i1, i2, p, exact)

