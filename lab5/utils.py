from math import floor, fabs, ceil, log10


def drange(start, stop, step):
    r = start
    while r < stop:
        n = floor(fabs(log10(0.02))) + 1
        yield r
        r += step
        r = ceil(r * 10 ** n) / 10 ** n
