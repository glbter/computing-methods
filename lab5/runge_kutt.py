import numpy as np


class RungeKutt:
    @staticmethod
    def runge_kutt(f, g, h: int, x0: int, y: int, z: int, xk: int) -> np.array:
        return np.asarray(RungeKutt._runge_kutt(f, g, h, x0, y, z, xk+h))[:, :2]

    @staticmethod
    def _runge_kutt(f, g, h: int, x0: int, y: int, z: int, xk: int) -> np.array:
        xyzgf = [(x0, y, z, y, z)]
        for x in np.arange(x0, xk-h, h):
            k1 = h * f(x, y, z)
            l1 = h * g(x, y, z)
            k2 = h * f(x + h / 2, y + k1 / 2, z + l1 / 2)
            l2 = h * g(x + h / 2, y + k1 / 2, z + l1 / 2)
            k3 = h * f(x + h / 2, y + k2 / 2, z + l2 / 2)
            l3 = h * g(x + h / 2, y + k2 / 2, z + l2 / 2)
            k4 = h * f(x + h, y + k3, z + l3)
            l4 = h * g(x + h, y + k3, z + l3)

            dy = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            dz = 1 / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
            y += dy
            z += dz
            xyzgf.append((x + h, y, z, g(x+h, y, z), f(x+h, y, z)))
        return xyzgf
