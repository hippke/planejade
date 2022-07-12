import numpy as np
import matplotlib.pyplot as plt
#import pandoramoon as pandora
from planejade.grids import timegrid
from planejade_caller import lc, fit
from wotan import flatten
import time as ttime
from planejade.helpers import ld_convert, ld_invert
import time as ttime
import fastrand
from numba import jit



# Create Pandora time grid and planet+moon model flux
# We use a more functional approach here to ease feeding the posterior results 
# into fresh light curves
epoch_duration = 365.25
t0_bary = 11
model_time = timegrid(
    t0_bary=t0_bary, 
    epochs=3, 
    epoch_duration=epoch_duration, 
    cadences_per_day=48, 
    epoch_distance=365.25,
    supersampling_factor=1
    ) #+ epoch_duration/2 - 10
#model_time = np.linspace(0, 800, 10000)
#model_time = np.linspace(0, 1000, 20000)
#print("Number of datapoints", len(model_time))


# The parameter names are not used, but written down for clarity
model_param_names = ["per", "a", "r", "b", "ecc", "w", "t0", "u1", "u2", "time"]

R_sun = 696_342_000
M_earth = 6e24
M_jup = 1.9e27
M_ganymede = 1.5e23

# r_planet=0.04  ==> SDE 10.5, SNR 9.1 - both OK
# r_planet=0.037 ==> SDE  9.2, SNR 7.8 - both OK
# r_planet=0.035 ==> SDE  8.3, SNR 7.0 - both IK TLS OK (300-400d)
# r_planet=0.032 ==> SDE  5.7, SNR 5.9 - 
# ensemble vs. single run n_pop=2000, probability? try experiments

# Create light curve based on the following parameter values and the time grid
model_param_values = np.array([365.25, 215, 0.09, 0.3, 0, 0, 11.33, 0.4089, 0.2556])
model_flux = lc(model_param_values, model_time)


from wotan.slider import running_segment
mask = np.ones(len(model_time))

flatten_lc = running_segment(model_time, model_flux, mask=mask, window_length=0.5, edge_cutoff=0, cval=5, method_code=1)

flatten_lc, trend_lc = flatten(model_time, model_flux, window_length=0.5, method='median', return_trend=True)

from numba import jit
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

def filt(x, N):
    return uniform_filter1d(x, size=N)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

from collections import deque
from bisect import insort, bisect_left
from itertools import islice
def running_median_insort(seq, window_size):
    """Contributed by Peter Otten"""
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def RunningMedian(x,N):
    idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    b = [row[row>0] for row in x[idx]]
    return np.array(map(np.median,b))
    #return np.array([np.median(c) for c in b])  # This also works
import bottleneck


def median_partial_detrender(x, N):
    out = np.copy(x)
    #out = runningMeanFast(x[100:500], N)
    #out = runningMeanFast(x[100:500], N)
    #out = runningMeanFast(x[100:500], N)
    #out = runningMeanFast(x[100:500], N)
    #out = runningMeanFast(x[100:500], N)

    bottleneck.move_median(x[100:500], N)
    bottleneck.move_median(x[100:500], N)
    bottleneck.move_median(x[100:500], N)
    bottleneck.move_median(x[100:500], N)
    bottleneck.move_median(x[100:500], N)

    #for idx in range(len(x)):
    #    if 1000 < idx < 2000:
    #        out[idx] = np.median(x[idx-5:idx+5])
    return out


import planejade as planejade
def log_likelihood(p, flux, time, yerr):
    f = planejade.planejade(per=p[0], a=p[1], r=p[2], b=p[3], ecc=0, w=0, 
        t0=p[4], u1=p[5], u2=p[6], time=time)
    loglike = -0.5 * np.nansum(((flux-f) / yerr)**2)
    return loglike


ftest = median_partial_detrender(model_flux, 31)


epoch_distance = 365.25
epoch_duration = 2

bounds = {
   "per": (1, 400),  # Period [days]
   #"t0":  (0, 1),    # Time of inferior conjunction [days]
   "M_star": (0.5, 1.5)
   #"a":   (2, 1000),      # Semi-major axis [stellar radii]
   #"b":   (0.2, 0.4),
   #"r":   (0.08, 0.1)
    }


noise_level = 1500e-6  # Gaussian noise to be added to the generated data
np.random.seed(0)  # Reproducibility
noise = np.random.normal(0, noise_level, len(model_time))
testdata = noise + model_flux
yerr = np.full(len(testdata), noise_level)
#cs = np.arange(start=0.01, stop=0.2, step=25)
cs = np.linspace(0.09, 0.12, num=1)
#a2 = np.arange(start=1000, stop=5000, step=10)
#arr_live_points = np.append(arr_live_points, a2)
#print(arr_live_points)
for c in cs:
    # New noise in every experiment to remove clustering of recoveries around certain #livepoints
    
    result_planet_only = fit(
        time=model_time, 
        flux=testdata, 
        yerr=yerr,
        bounds=bounds,
        moon=False,
        live_points=250,
        n_it=2000,
        batches=1,
        c=c
        )
    #true_model = lc(model_param_values, model_time)
    #logl_true = log_likelihood(model_param_values, testdata, model_time, yerr)
    #print(" logl_true", logl_true)

#print(result_planet_only["logl"])
# perfect: 26141


"""
recovered_flux = lc(result_planet_only["points"], model_time)

plt.clf()
plt.close()
polished = (testdata - recovered_flux) + 1
flatten_lc, trend_lc = flatten(model_time, polished, window_length=0.3, method='biweight', return_trend=True)
plt.scatter(model_time, flatten_lc, s=1, color="black")
plt.plot(model_time, model_flux, color="blue")
plt.plot(model_time, recovered_flux, color="red")

plt.show()
"""