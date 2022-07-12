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
    no_points = len(flux)#[0]

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

    #plt.close()
    #plt.clf()
    #plt.plot(np.linspace(0, len(rlist_verlauf[0]), len(rlist_verlauf[0])), rlist_verlauf[0])
    #plt.xscale("log")
    #plt.yscale("log")
    #plt.xlim(0, len(rlist_verlauf[0]))
    #plt.show()
    BIC = ndim * np.log(no_points) - 2 * best_logl
    AIC = 2 * ndim - 2 * best_logl
    best_popt = rlist_popt[np.argmax(rlist_ll)]

    # Convert phase to T0
    phase = best_popt[4]
    phase = best_popt[4]
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
    


    result_dict = {
       "logl": best_logl,
       "BIC": BIC,
       "AIC": AIC,
       "points": best_popt
       }
    return result_dict
