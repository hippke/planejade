import numpy as np
from numpy import sqrt, pi, arccos, abs, log, ceil
from numba import jit


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def cci(r1, r2, d):
    """Circle-Circle-Intersect to calculate the area of asymmetric "lens"
    Source: http://mathworld.wolfram.com/Circle-CircleIntersection.html"""
    if r1 < d - r2:
        return 0
    elif r1 >= d + r2:
        return pi * r2**2
    elif d - r2 <= -r1:
        return pi * r1**2
    else:
        return (
            r2**2 * arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
            + r1**2 * arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
            - 0.5 * sqrt((-d + r2 + r1) * (d + r2 - r1) * (d - r2 + r1) * (d + r2 + r1))
        )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_small_single_value(z, k, u1, u2):
    """Small body approximation by Mandel-Agol. Adequate for k<~0.01
    This version does not calculate an entire array, but just one value.
    It is used by occult_hybrid with its linear interpolation between exact values
    and small-planet approximation."""
    s = 2 * pi * 1 / 12 * (-2 * u1 - u2 + 6)
    m = sqrt(1 - min(z**2, 1))
    limb_darkening = 1 - u1 * (1 - m) - u2 * (1 - m) ** 2
    area = cci(1, k, z)
    flux = (s - limb_darkening * area) * (1 / s)

    # Some combinations of (u1,u2) cause flux>1 which is unphysical
    if flux > 1:
        flux = 1
    return flux


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellpicb(n, k):
    """The complete elliptical integral of the third kind
    Bulirsch 1965, Numerische Mathematik, 7, 78
    Bulirsch 1965, Numerische Mathematik, 7, 353
    Adapted from L. Kreidbergs C version in BATMAN
    (Kreidberg, L. 2015, PASP 957, 127)
    (https://github.com/lkreidberg/batman)
    which is translated from J. Eastman's IDL routine
    in EXOFAST (Eastman et al. 2013, PASP 125, 83)"""
    HALF_PI = 0.5 * pi
    kc = sqrt(1 - k**2)
    e = kc
    p = sqrt(n + 1)
    m0 = 1
    c = 1
    d = 1 / p
    for nit in range(1000):
        f = c
        c = d / p + c
        g = e / p
        d = 2 * (f * g + d)
        p = g + p
        g = m0
        m0 = kc + m0
        if abs(1 - kc / g) > 1e-8:
            kc = 2 * sqrt(e)
            e = kc * m0
        else:
            return HALF_PI * (c * m0 + d) / (m0 * (m0 + p))
    return 0


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellec(k):
    a1 = 0.443251414630
    a2 = 0.062606012200
    a3 = 0.047573835460
    a4 = 0.017365064510
    b1 = 0.249983683100
    b2 = 0.092001800370
    b3 = 0.040696975260
    b4 = 0.005264496390
    m1 = 1 - k * k
    epsilon = 1e-14
    return (1 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))) + (
        m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * log(1 / (m1 + epsilon))
    )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellk(k):
    a0 = 1.386294361120
    a1 = 0.096663442590
    a2 = 0.035900923830
    a3 = 0.037425637130
    a4 = 0.014511962120
    b0 = 0.50
    b1 = 0.124985935970
    b2 = 0.068802485760
    b3 = 0.033283553460
    b4 = 0.004417870120
    m1 = 1 - k * k
    epsilon = 1e-14
    return (a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))) - (
        (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * log(m1 + epsilon)
    )



@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_hybrid(zs, k, u1, u2):
    """Evaluates the transit model for an array of normalized distances.
    This version performs linear interpolation between exact values and small-planet.

    Parameters
    ----------
    z: 1D array
        Normalized distances
    k: float
        Planet-star radius ratio
    u1, u2: float
        Limb darkening coefficients
    Returns
    -------
    Transit model evaluated at `z`.
    """
    if abs(k - 0.5) < 1e-4:
        k = 0.5

    interpol_flux_1 = 0
    interpol_flux_2 = 0

    INV_PI = 1 / pi
    k2 = k**2
    flux = np.empty(len(zs))
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2

    for i in range(-2, len(zs)):

        # Linear interpolation method: First two values are to interpolate
        if i == -2:
            z = 0
        elif i == -1:
            z = 0.65
        else:
            z = zs[i]

        if abs(z - k) < 1e-6:
            z += 1e-6

        # Source is unocculted
        if z > 1 + k or z < 0:
            flux[i] = 1
            continue

        z2 = z**2
        x1 = (k - z) ** 2
        x2 = (k + z) ** 2
        x3 = k**2 - z2

        # Star partially occulted and the occulting object crosses the limb
        if z >= abs(1 - k) and z <= 1 + k:
            kap1 = arccos(min((1 - k2 + z2) / (2 * z), 1))
            kap0 = arccos(min((k2 + z2 - 1) / (2 * k * z), 1))
            lex = k2 * kap0 + kap1
            lex = (lex - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI

        # Occulting object transits the source star (but doesn't completely cover it):
        if z <= 1 - k:
            lex = k2

        # Occulting body partly occults source and crosses the limb: Case III:
        if (z > 0.5 + abs(k - 0.5) and z < 1 + k) or (
            k > 0.5 and z > abs(1 - k) and z < k
        ):
            q = sqrt((1 - (k - z) ** 2) / 4 / z / k)
            ldx = (
                1
                / 9
                * INV_PI
                / sqrt(k * z)
                * (
                    ((1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2)) * ellk(q)
                    + 4 * k * z * (z2 + 7 * k2 - 4) * ellec(q)
                    - 3 * x3 / x1 * ellpicb((1 / x1 - 1), q)
                )
            )
            if z < k:
                ldx = ldx + 2 / 3
            edx = (
                1
                / 2
                * INV_PI
                * (
                    kap1
                    + k2 * (k2 + 2 * z2) * kap0
                    - (1 + 5 * k2 + z2) / 4 * sqrt((1 - x1) * (x2 - 1))
                )
            )

        # Use interpolation method
        if (
            (i >= 0 and k <= 0.05 and z <= 0.65)
            or (i >= 0 and k <= 0.04 and z <= 0.70)
            or (i >= 0 and k <= 0.03 and z <= 0.80)
            or (i >= 0 and k <= 0.02 and z <= 0.95)
            or (i >= 0 and k <= 0.01 and z <= 0.98)
        ):
            # Perform linear interpolation correction
            testv = occult_small_single_value(z, k, u1, u2)
            flux[i] = (
                occult_small_single_value(z, k, u1, u2)
                + interpol_flux_1
                + interpol_flux_2 * z
            )
            continue

        # Occulting body transits the source: Table 3, Case IV:
        if z < (1 - k):
            q = sqrt((x2 - x1) / (1 - x1))
            ldx = (
                2
                / 9
                * INV_PI
                / sqrt(1 - x1)
                * (
                    (1 - 5 * z2 + k2 + x3 * x3) * ellk(q)
                    + (1 - x1) * (z2 + 7 * k2 - 4) * ellec(q)
                    - 3 * x3 / x1 * ellpicb((x2 / x1 - 1), q)
                )
            )
            if z < k:
                ldx = ldx + 2 / 3
            if abs(k + z - 1) < 1e-4:
                ldx = 2 / 3 * INV_PI * arccos(1 - 2 * k) - 4 / 9 * INV_PI * sqrt(
                    k * (1 - k)
                ) * (3 + 2 * k - 8 * k2)
            edx = k2 / 2 * (k2 + 2 * z2)

        current_flux = 1 - (c1 * lex + c2 * ldx + u2 * edx) * omega

        # Linear interpolation method: First two values are to interpolate
        if i == -2:
            interpol_flux_1 = current_flux - occult_small_single_value(z, k, u1, u2)
        elif i == -1:
            interpol_flux_2 = current_flux - occult_small_single_value(z, k, u1, u2)
        else:
            flux[i] = current_flux

        # Some combinations of (u1,u2) cause flux>1 which is unphysical
        if flux[i] > 1:
                flux[i] = 1

    return flux
