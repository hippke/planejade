import numpy as np
import planejade as planejade
from planejade.helpers import ld_convert, ld_invert
import pyfde
from numba import jit, prange
import random
import os
from tqdm import tqdm
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
import ultranest
import ultranest.stepsampler
from ultranest import ReactiveNestedSampler
import lmfit
from scipy.ndimage.filters import uniform_filter1d
import matplotlib.pyplot as plt
from wotan import flatten
from jade_mod_solo import JADE
from tabulate import tabulate


def count_stats(t, y, transit_times, transit_duration_in_days):
    """Return:
    * in_transit_count:     Number of data points in transit (phase-folded)
    * after_transit_count:  Number of data points in a bin of transit duration, 
                            after transit (phase-folded)
    * before_transit_count: Number of data points in a bin of transit duration, 
                            before transit (phase-folded)
    """
    in_transit_count = 0
    after_transit_count = 0
    before_transit_count = 0

    for mid_transit in transit_times:
        T0 = (
            mid_transit - 1.5 * transit_duration_in_days
        )  # start of 1 transit dur before ingress
        T1 = mid_transit - 0.5 * transit_duration_in_days  # start of ingress
        T4 = mid_transit + 0.5 * transit_duration_in_days  # end of egress
        T5 = (
            mid_transit + 1.5 * transit_duration_in_days
        )  # end of egress + 1 transit dur

        if T0 > min(t) and T5 < max(t):  # inside time
            idx_intransit = np.where(np.logical_and(t > T1, t < T4))
            idx_before_transit = np.where(np.logical_and(t > T0, t < T1))
            idx_after_transit = np.where(np.logical_and(t > T4, t < T5))
            points_in_this_in_transit = len(y[idx_intransit])
            points_in_this_before_transit = len(y[idx_before_transit])
            points_in_this_after_transit = len(y[idx_after_transit])

            in_transit_count += points_in_this_in_transit
            before_transit_count += points_in_this_before_transit
            after_transit_count += points_in_this_after_transit

    return in_transit_count, after_transit_count, before_transit_count


def intransit_stats(t, y, transit_times, transit_duration_in_days):
    """Return all intransit odd and even flux points"""

    all_flux_intransit_odd = np.array([])
    all_flux_intransit_even = np.array([])
    all_flux_intransit = np.array([])
    all_idx_intransit = np.array([])
    per_transit_count = np.zeros([len(transit_times)])
    transit_depths = np.zeros([len(transit_times)])
    transit_depths_uncertainties = np.zeros([len(transit_times)])

    for i in range(len(transit_times)):

        depth_mean_odd = np.nan
        depth_mean_even = np.nan
        depth_mean_odd_std = np.nan
        depth_mean_even_std = np.nan

        mid_transit = transit_times[i]
        tmin = mid_transit - 0.5 * transit_duration_in_days
        tmax = mid_transit + 0.5 * transit_duration_in_days
        if np.isnan(tmin) or np.isnan(tmax):
            idx_intransit = []
            flux_intransit = []
            mean_flux = np.nan
        else:
            idx_intransit = np.where(np.logical_and(t > tmin, t < tmax))
            flux_intransit = y[idx_intransit]
            if len(y[idx_intransit]) > 0:
                mean_flux = np.mean(y[idx_intransit])
            else:
                mean_flux = np.nan
        intransit_points = np.size(y[idx_intransit])
        transit_depths[i] = mean_flux
        if len(y[idx_intransit] > 0):
            transit_depths_uncertainties[i] = np.std(y[idx_intransit]) / np.sqrt(
                intransit_points
            )
        else:
            transit_depths_uncertainties[i] = np.nan
        per_transit_count[i] = intransit_points

        # Check if transit odd/even to collect the flux for the mean calculations
        if i % 2 == 0:  # even
            all_flux_intransit_even = np.append(
                all_flux_intransit_even, flux_intransit
            )
        else:  # odd
            all_flux_intransit_odd = np.append(
                all_flux_intransit_odd, flux_intransit
            )
        if len(all_flux_intransit_odd) > 0:
            depth_mean_odd = np.mean(all_flux_intransit_odd)

            depth_mean_odd_std = np.std(all_flux_intransit_odd) / np.sum(
                len(all_flux_intransit_odd)
            ) ** (0.5)
        if len(all_flux_intransit_even) > 0:
            depth_mean_even = np.mean(all_flux_intransit_even)
            depth_mean_even_std = np.std(all_flux_intransit_even) / np.sum(
                len(all_flux_intransit_even)
            ) ** (0.5)

    return (
        depth_mean_odd,
        depth_mean_even,
        depth_mean_odd_std,
        depth_mean_even_std,
        all_flux_intransit_odd,
        all_flux_intransit_even,
        per_transit_count,
        transit_depths,
        transit_depths_uncertainties,
    )


def snr_stats(
    t,
    y,
    period,
    duration,
    T0,
    transit_times,
    transit_duration_in_days,
    per_transit_count,
):
    """Return snr_per_transit and snr_pink_per_transit"""

    snr_per_transit = np.zeros([len(transit_times)])
    snr_pink_per_transit = np.zeros([len(transit_times)])
    intransit = transit_mask(t, period, 2 * duration, T0)
    flux_ootr = y[~intransit]

    try:
        pinknoise = pink_noise(flux_ootr, int(np.mean(per_transit_count)))
    except:
        pinknoise = np.nan

    # Estimate SNR and pink SNR
    # Second run because now the out of transit points are known
    if len(flux_ootr) > 0:
        std = np.std(flux_ootr)
    else:
        std = np.nan
    for i in range(len(transit_times)):
        mid_transit = transit_times[i]
        tmin = mid_transit - 0.5 * transit_duration_in_days
        tmax = mid_transit + 0.5 * transit_duration_in_days
        if np.isnan(tmin) or np.isnan(tmax):
            idx_intransit = []
            mean_flux = np.nan
        else:
            idx_intransit = np.where(np.logical_and(t > tmin, t < tmax))
            if len(y[idx_intransit]) > 0:
                mean_flux = np.mean(y[idx_intransit])
            else:
                mean_flux = np.nan

        intransit_points = np.size(y[idx_intransit])
        try:
            snr_pink_per_transit[i] = (1 - mean_flux) / pinknoise
            if intransit_points > 0 and not np.isnan(std):
                std_binned = std / intransit_points ** 0.5
                snr_per_transit[i] = (1 - mean_flux) / std_binned
            else:
                snr_per_transit[i] = 0
                snr_pink_per_transit[i] = 0
        except:
            snr_per_transit[i] = 0
            snr_pink_per_transit[i] = 0

    return snr_per_transit, snr_pink_per_transit


def all_transit_times(t0, t, period):
    """Return all mid-transit times within t"""

    if t0 < min(t):
        transit_times = [t0 + period]
    else:
        transit_times = [t0]
    previous_transit_time = transit_times[0]
    transit_number = 0
    while True:
        transit_number = transit_number + 1
        next_transit_time = previous_transit_time + period
        if next_transit_time < (np.min(t) + (np.max(t) - np.min(t))):
            transit_times.append(next_transit_time)
            previous_transit_time = next_transit_time
        else:
            break
    return transit_times



@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def semimajor_axis(period, M_star):
    """Takes planetary period [days] and stellar mass [solar masses]
    Return semimajor axis in stellar radii (=solar radii), 
    # assuming unknown stellar density as solar density
    """
    # Naively we would fit for free semimajor axis a=[2, 1000 R_star] 
    # and free period period=[1, 1000 days]. But per and a are related by Kepler's 3rd law
    # Thus we fit for M_star=(M_star+M_planet) and calculate a=(per^(2/3) / M_star^(-1/3)))
    days_per_year = 365.25
    R_sun = 696_342_000  # m
    au = 149_597_870_700  # m
    a = (period / days_per_year)**(2/3) / M_star**(-1/3)
    a *= (au / R_sun)
    return a

#per = 365.25  # days
#M = 1  # M_sun

#print("a", a)
#print("a", semimajor_axis(per, M))



#@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def lc(p, time):
    if len(p) == 7:
        f = planejade.planejade(per=p[0], a=p[1], r=p[2], b=p[3], ecc=0, w=0, 
            t0=p[4], u1=p[5], u2=p[6], time=time)
    else:
        f = planejade.planejade(per=p[0], a=p[1], r=p[2], b=p[3], ecc=p[4], w=p[5], 
            t0=p[6], u1=p[7], u2=p[8], time=time)
    return f


def fit_emcee(time, flux, yerr, bounds):


    def residuals(p):
        N = 100
        v = p.valuesdict()
        q1 = v["q1"]
        q2 = v["q2"]
        u1, u2 = ld_convert(q1, q2)
        f = planejade.planejade(per=v["per"], a=v["a"], r=v["r"], b=v["b"], ecc=0, w=0, 
            t0=v["t0"], u1=u1, u2=u2, time=time)
        trend = uniform_filter1d(f, N)
        f = f / trend
        residuals = abs(flux - f) / yerr
        #print("Eval done")
        return residuals

    


    params = lmfit.Parameters()
    params.add('per', value=365.25, min=1, max=1000)
    params.add('a', value=215, min=2, max=2000)
    params.add('r', value=0.08, min=0.01, max=0.12)
    params.add('b', value=0.3, min=0., max=1.)
    params.add('t0', value=11.33, min=0, max=200)
    params.add('q1', value=0.4089, min=0., max=1.)
    params.add('q2', value=0.2556, min=0., max=1.)

    result = lmfit.minimize(
        residuals,
        params=params,
        method="emcee",
        #nan_policy='omit', 
        burn=1000, 
        steps=3000,
        nwalkers=100,
        #thin=thin,
        #is_weighted=True,
        progress=True,
        #workers=8
        )

    return result    


def fit_ultranest(time, flux, yerr, bounds, live_points):

    #@jit(cache=True, nopython=True, fastmath=True, parallel=False)
    def log_likelihood(p):
        q1 = p[5]
        q2 = p[6]
        u1, u2 = ld_convert(q1, q2)
        f = planejade.planejade(per=p[0], a=p[1], r=p[2], b=p[3], ecc=0, w=0, 
            t0=p[4], u1=u1, u2=u2, time=time)
        loglike = -0.5 * np.nansum(((f - flux) / yerr)**2)
        return loglike

    def prior_transform(cube):
        p = cube.copy()
        for idx, key in enumerate(bounds):
            low, high = bounds[idx]
            p[idx]  = cube[idx]  * (high - low) + low
        return p

    bounds = list(bounds.values())
    sampler = ReactiveNestedSampler(
        param_names=["per", "a", "r", "b", "t0", "q1", "q2"],
        loglike=log_likelihood, 
        transform=prior_transform,
        )
    #sampler.stepsampler = ultranest.stepsampler.SliceSampler(
    #    nsteps=4000,
    #    adaptive_nsteps='move-distance',
    #    generate_direction=ultranest.stepsampler.generate_cube_oriented_differential_direction
    #    )  
    result = sampler.run(min_num_live_points=live_points)
    sampler.print_results()
    return result


def fit(time, flux, yerr, n_it, bounds=None, moon=True, live_points=750, batches=64, c=0.1):

    @jit(cache=True, nopython=True, fastmath=True, parallel=False)
    def energy(p):
        return np.sqrt(-2 * log_likelihood(p) / (nobs - ndim))

    @jit(cache=True, nopython=True, fastmath=True, parallel=False)
    def log_likelihood(p):
        N = 25
        u1 = p[5]
        u2 = p[6]
        phase = p[4]
        period = p[0]
        t0 = phase * period
        M_star = p[1]
        a = semimajor_axis(period, M_star)
        q1 = p[5]
        q2 = p[6]
        u1, u2 = ld_convert(q1, q2)
        #print("values:", p)
        f = planejade.planejade(per=period, a=a, r=p[2], b=p[3], ecc=0, w=0, 
            t0=t0, u1=u1, u2=u2, time=time)
        """
        polished = (flux - f) + 1
        
        flatten_lc = flatten(
            time, 
            polished, 
            window_length=1, 
            method='biweight', 
            return_trend=False
            )
        
        loglike = -0.5 * np.nansum(((flatten_lc-1) / yerr)**2)
        """
        loglike = -0.5 * np.nansum(((flux-f) / yerr)**2)
        #loglike = -np.sum(((flux - f) / yerr)**2)
        #print(loglike)
        #print("loglike", loglike)
        #print(p[0], loglike)
        return loglike

    def run_one_batch(batch_id):
        # Need to re-seed each function call inside pool
        #np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        """
        solver = pyfde.JADE(
            energy,
            n_dim=ndim,
            n_pop=1000,
            limits=bounds,
            seed=np.random.randint(1, 2147483647), 
            minimize=True
            )
        """

        


        solver = JADE(
            log_likelihood,
            n_dim=ndim,
            n_pop=live_points,
            limits=bounds,
            n_it=n_it,
            batch_id=batch_id,
            c=c
            )
        
        #solver.c = 0.01    # adaption parameter. Default 0.1. About 1.6x better to use 0.01
        iterations = n_it  # 2000 is a good trade-off
        result, verlauf, list_logls, list_periods = solver.run(n_it=iterations)
        #result = popt[0]
        
        evals = popsize * iterations
        return (log_likelihood(result), result, evals, verlauf, list_logls, list_periods)

    

    # Bounds for planet-only model
    planet_bounds = {
       "per":   (1, 1000),                  # Period [days]
       #"a":     (2, 1000),                  # Semi-major axis [stellar radii]
       "M_star": (0.1,3.0),
       "r":   (0.01, 0.12),               # Planet radius [stellar radii]
       "b":     (0, 1),                     # Impact parameter [0,1]
       #"ecc": (0, 1),
       #"w": (0, 180),
       "t0":    (0, 1),                  # Time of inferior conjunction [days]
       "u1":         (0, 1),
       "u2":         (0, 1)

    }

    planet_bounds.update(bounds)
    bounds = list(planet_bounds.values())
    #print("bounds", bounds)
    ndim = 7

    # BIC baseline
    no_points = len(flux)
    loglike_baseline = -0.5 * np.nansum(((flux - 1) / yerr)**2)
    print(loglike_baseline, "loglike base model")
    BIC_baseline = 1 * np.log(no_points) - 2 * (loglike_baseline)
    print("BIC_baseline", BIC_baseline)
    print("points", no_points)

    popsize = int(live_points / ndim)  # popsize is live_points / ndim e.g. 750/15=50
    nobs = len(flux)
    cores = int(multiprocessing.cpu_count())
    #print("Search running. Wait for first update. Using all CPU threads:", cores)
    rlist_ll = []
    rlist_popt = []
    rlist_evals = []
    rlist_verlauf = []
    list_logls = []
    list_periods = []
    batch_list = np.linspace(1, batches, batches, dtype="int")
    pool = Pool(cores)
    pool.restart()
    #pbar = tqdm(total=batches, smoothing=0)
    for data in pool.imap(run_one_batch, batch_list):
        rlist_ll.append(data[0])
        rlist_popt.append(data[1])
        rlist_evals.append(data[2])
        rlist_verlauf.append(data[3])
        list_logls.append(data[4])
        list_periods.append(data[5])
        #pbar.update(1)
        #pbar.set_postfix({'ncall': np.sum(rlist_evals), 'logl': str(round(np.max(rlist_ll), 2))})
    pool.close()
    pool.join() 
    #pbar.close()
    best_logl = max(rlist_ll)
    

    #print(list_logls)
    #print(list_logls)
    #list_periods = [x for xs in list_periods for x in xs]
    #list_logls = [x for xs in list_logls for x in xs]
    #list_logls = np.array(list_logls).flatten()
    #list_periods = np.array(list_periods).flatten()
    #list_periods = np.sort(list_periods)
    #arr1inds = list_periods.argsort()
    #list_logls = list_logls[arr1inds]
    list_logls = [x for xs in list_logls for x in xs]
    list_logls = np.array(list_logls)
    list_logls = list_logls.flatten()

    list_periods = [x for xs in list_periods for x in xs]
    list_periods = np.array(list_periods)
    list_periods = list_periods.flatten()

    #list_periods = np.sort(list_periods)
    arr1inds = list_periods.argsort()
    list_logls = list_logls[arr1inds]
    list_periods = list_periods[arr1inds]
    """
    print(len(list_logls))
    print(max(list_logls))
    print(min(list_logls))

    step = 0.1
    period_grid = np.arange(1, 400, step=step)
    logl_grid = np.zeros(len(period_grid))
    for idx, period in enumerate(period_grid):
        start = period - step/2
        stop = period + step/2
        indexes = np.where((list_periods > start) & (list_periods < stop))
        if len(list_logls[indexes]) > 0:
            current_max = np.max(list_logls[indexes])
        else:
            current_max = 0#np.nan
        logl_grid[idx] = current_max
        print(idx, period, start, stop, current_max)

    # Scale logs
    logl_grid += np.median(logl_grid)
    #logl_grid /= np.nanmax(logl_grid)

    plt.plot(period_grid, logl_grid)
    #plt.yscale("log")
    plt.show()

    #print(list_logls)
    plt.plot(list_periods, list_logls)
    plt.show()


    #list_logls = list_logls.ravel()  #[x for xs in list_logls for x in xs]
    #list_periods = [x for xs in list_periods for x in xs]


    #print(list_logls)
    """
    #plt.close()
    #plt.clf()
    #plt.plot(np.linspace(0, len(rlist_verlauf[0]), len(rlist_verlauf[0])), rlist_verlauf[0])
    #plt.xscale("log")
    #plt.yscale("log")
    #plt.xlim(0, len(rlist_verlauf[0]))
    #plt.show()



    BIC = ndim * np.log(no_points) - 2 * best_logl
    print(best_logl, "best_logl")
    print(round(BIC_baseline, 1), "BIC_baseline")
    print(round(BIC, 1), "BIC")
    #AIC = 2 * ndim - 2 * best_logl
    best_popt = rlist_popt[np.argmax(rlist_ll)]
    deltaBIC = BIC_baseline - BIC
    print(round(deltaBIC, 1), "deltaBIC")
    if deltaBIC > 10:
        print("Strong evidence for the planet model")
    if deltaBIC < 0:
        print("Negative evidence for the planet model")

    # Try model with half the period
    p_half = best_popt.copy()
    p_half[0] /= 2
    lc_half = log_likelihood(p_half)
    BIC_lc_half = ndim * np.log(no_points) - 2 * lc_half
    print(BIC_lc_half, "BIC half period")
    if BIC_lc_half < BIC:
        print("Model with half the period is better")

    # Try model with half the period, shifted by half phase
    p_half = best_popt.copy()
    p_half[0] /= 2
    p_half[4] += 0.5
    if p_half[4] > 1:
        p_half[4] -= 1
    lc_half = log_likelihood(p_half)
    BIC_lc_half = ndim * np.log(no_points) - 2 * lc_half
    print(BIC_lc_half, "BIC half period")
    if BIC_lc_half < BIC:
        print("Model with half the period (shifted half phase) is better")

    # Try model with twice the period
    p_twice = best_popt.copy()
    p_twice[0] *= 2
    lc_twice = log_likelihood(p_twice)
    BIC_lc_twice = ndim * np.log(no_points) - 2 * lc_twice
    print(BIC_lc_twice, "BIC twice period")
    if BIC_lc_twice < BIC:
        print("Model with twice the period is better")


    # Convert phase to T0
    phase = best_popt[4]
    #phase = best_popt[4]
    period = best_popt[0]
    t0 = phase * period
    while t0 < min(time):
        t0 += period
    best_popt[4] = t0
    #print("phase", phase)
    #print("t0", t0)

    # Convert M_star to semimajor axis a
    M_star = best_popt[1]
    a = semimajor_axis(period, M_star)
    best_popt[1] = a
    #print("M_star", M_star)
    #print("a", a)
    q1 = best_popt[5]
    q2 = best_popt[6]
    u1, u2 = ld_convert(q1, q2)
    best_popt[5] = u1
    best_popt[6] = u2

    # SNR calculation
    in_transit_mask = np.where(lc(best_popt, time) < 1)
    in_transit_points_data = len(flux[in_transit_mask])
    S = np.mean(1-flux[in_transit_mask])
    N = np.mean(yerr[in_transit_mask]) / np.sqrt(in_transit_points_data)
    SNR = S / N
    # SNR is wrong: should scale linearly with number of transits (?)

    print("Maximum Likelihood Estimation from Ensemble-JADE")
    headers = ["Parameter", "Name", "Unit", "MLE value"]
    table = [
        ["per", "Period",                       "days", round(best_popt[0], 5)],
        ["a",   "Semi-major axis",              "R٭",   round(best_popt[1], 5)],
        ["r",   "Radius",                       "R٭",   round(best_popt[2], 5)],
        ["b",   "Impact parameter",             1,      round(best_popt[3], 5)],
        ["t0",  "Time of inferior conjunction", "days", round(best_popt[4], 5)],
        ["u1",  "Quadratic limb-darkening u1",  1,      round(best_popt[5], 5)],
        ["u2",  "Quadratic limb-darkening u2",  1,      round(best_popt[6], 5)],
        ["ecc", "Eccentricity",                 1,      "0 (fixed)"],
        ["w",   "Argument of periapsis",        "deg",  "0 (fixed)"]
        ]
    print(tabulate(table, headers=headers))
    print("")
    headers = ["Model statistic", "Value", "Comment"]
    if deltaBIC < 0:
        BIC_comment = "No"
    elif deltaBIC >= 0 and deltaBIC < 2.3:
        BIC_comment = "Weak"
    elif deltaBIC >=2.3 and deltaBIC < 4.61:
        BIC_comment = "Substantial"
    elif deltaBIC >=4.61 and deltaBIC < 6.91:
        BIC_comment = "Strong"
    elif deltaBIC >=6.91 and deltaBIC < 9.21:
        BIC_comment = "Very strong"
    else: 
        BIC_comment = "Decisive"
    BIC_comment += " evidence for the planet model"

    if SNR < 7.1:
        SNR_comment = "<7.1: not significant (Borucki+ 2011)"
    elif SNR >= 7.1 and SNR < 10:
        SNR_comment = ">7.1: significant (Borucki+ 2011)"
    else:
        SNR_comment = ">10: highly significant (Howard+ 2011)"

    table = [
        ["Minimum log-likelihood",      round(best_logl, 1)],
        ["Reduced chi²"],
        ["ΔBIC",                        round(deltaBIC, 1), BIC_comment],
        ["Signal-to-noise ratio (SNR)", round(SNR, 1),      SNR_comment],
        ["Odd-even mismatch",         "X.Xσ"],
        ["Transit count (with data)", "X (X)"]
        ]
    print(tabulate(table, headers=headers))

    transit_times = all_transit_times(t0=best_popt[4], t=time, period=best_popt[0])
    print(transit_times)

    per = best_popt[0]
    a = best_popt[1]
    tdur = per / np.pi * np.arcsin(1 / a)
    print("tdur", tdur)
    #if ecc_bary > 0:
    #    tdur /= 1 / sqrt(1 - ecc_bary ** 2) * (1 + ecc_bary * cos((w_bary - 90) / 180 * pi))

    depth_mean_odd, depth_mean_even, depth_mean_odd_std, depth_mean_even_std, all_flux_intransit_odd, all_flux_intransit_even, per_transit_count, transit_depths, transit_depths_uncertainties = intransit_stats(time, flux, transit_times, tdur)
    print(depth_mean_odd,
        depth_mean_even,
        depth_mean_odd_std,
        depth_mean_even_std,
        all_flux_intransit_odd,
        all_flux_intransit_even,
        per_transit_count,
        transit_depths,
        transit_depths_uncertainties)

    # Odd even mismatch in standard deviations
    odd_even_difference = abs(depth_mean_odd - depth_mean_even)
    odd_even_std_sum = depth_mean_odd_std + depth_mean_even_std
    odd_even_mismatch = odd_even_difference / odd_even_std_sum
    print("odd_even_mismatch", odd_even_mismatch)

    """
    snr_per_transit, snr_pink_per_transit = 
        snr_stats(
        time,
        flux,
        period=per,
        duration,
        T0,
        transit_times,
        transit_duration_in_days,
        per_transit_count,
    """

    result_dict = {
       "logl": best_logl,
       "BIC": BIC,
       "points": best_popt
       }
    return result_dict
