from utils import drange
from runge_kutt import *


class AdamsMethods:
    @staticmethod
    def adams_method(f, g, h: float, x0: int, y: int, z: int, xk: int) -> np.array:
        xyzgf = RungeKutt._runge_kutt(f, g, h, x0, y, z, x0+3*h)

        for x in drange(x0 + 4*h, xk+h, h):
            yk, zk, fk1, gk1 = AdamsMethods._predictor_step(f, g, x, xyzgf, h)
            xyzgf.append((x, yk, zk, gk1, fk1))
        return np.asarray(xyzgf)[:, :2]

    @staticmethod
    def adams_boshfort_method(f, g, h: int, x0: int, y: int, z: int, xk: int) -> np.array:
        xyzgf = RungeKutt._runge_kutt(f, g, h, x0, y, z, x0+3*h)

        for x in drange(x0 + 4*h, xk+h, h):
            yk_pred, zk_pred, fk1, gk1 = AdamsMethods._predictor_step(f, g, x, xyzgf, h)
            yk, zk, fk1, gk1 = AdamsMethods._corrector_step(fk1, gk1, xyzgf, h, f, g, x)
            xyzgf.append((x, yk, zk, gk1, fk1))
        return np.asarray(xyzgf)[:, :2]

    @staticmethod
    def _predictor_step(f, g, x, xyzgf, h):
        fk = gk = xyzgf
        yk = xyzgf[-1][1] + h/24 * (55*fk[-1][-1] - 59*fk[-2][-1] + 37*fk[-3][-1] - 9*fk[-4][-1])
        zk = xyzgf[-1][2] + h/24 * (55*gk[-1][-2] - 59*gk[-2][-2] + 37*gk[-3][-2] - 9*gk[-4][-2])
        fk1 = f(x, yk, zk)
        gk1 = g(x, yk, zk)
        return yk, zk, fk1, gk1

    @staticmethod
    def _corrector_step(fk1, gk1, xyzgf, h, f, g, x):
        fk = gk = xyzgf
        yk = xyzgf[-1][1] + h/24 * (9*fk1 + 19*fk[-1][-1] - 5*fk[-2][-1] + fk[-3][-1])
        zk = xyzgf[-1][2] + h/24 * (9*gk1 + 19*gk[-1][-2] - 5*gk[-2][-2] + gk[-3][-2])
        fk1 = f(x, yk, zk)
        gk1 = g(x, yk, zk)
        return yk, zk, fk1, gk1

