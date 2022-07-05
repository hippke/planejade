import numpy as np
from numpy import sqrt, pi, arcsin, cos
from numba import jit
from planejade.occult import occult_hybrid

@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def planejade(per, a, r, b, ecc, w, t0, u1, u2, time):
    z = np.empty(len(time))
    tdur = per / pi * arcsin(1 / a)
    if ecc > 0:
        tdur /= 1 / sqrt(1 - ecc ** 2) * (1 + ecc * cos((w - 90) / 180 * pi))
    for idx in range(len(time)):
        epoch = int((time[idx] - t0) / per + 0.5)
        x = (2 * (epoch * per - time[idx] + t0)) / tdur
        z[idx] = sqrt(x ** 2 + b ** 2)  # 20% faster to do for all than "if x < 1.2...""
    f = occult_hybrid(zs=z, u1=u1, u2=u2, k=r)
    return f
