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
print("Number of datapoints", len(model_time))


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
model_param_values = np.array([365.25, 215, 0.035, 0.3, 0, 0, 11.33, 0.4089, 0.2556])
model_flux = lc(model_param_values, model_time)


from wotan.slider import running_segment
mask = np.ones(len(model_time))

flatten_lc = running_segment(model_time, model_flux, mask=mask, window_length=0.5, edge_cutoff=0, cval=5, method_code=1)

flatten_lc, trend_lc = flatten(model_time, model_flux, window_length=0.5, method='median', return_trend=True)
for i in range(10):
    t1 = ttime.perf_counter()
    #flatten_lc, trend_lc = flatten(model_time, model_flux, window_length=0.5, method='mean', return_trend=True)
    flatten_lc = running_segment(model_time, model_flux, mask=mask, window_length=0.5, edge_cutoff=0, cval=5, method_code=1)

    t2 = ttime.perf_counter()
    print("Runtime detrending", t2-t1)

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


ftest = median_partial_detrender(model_flux, 31)


plt.plot(model_time, model_flux, linewidth=1, color="blue")
plt.show()

noise_level = 1500e-6  # Gaussian noise to be added to the generated data
np.random.seed(0)  # Reproducibility
noise = np.random.normal(0, noise_level, len(model_time))
testdata = noise + model_flux
yerr = np.full(len(testdata), noise_level)

#print(model_time)
trend = ((np.sin(model_time) + model_time / 10 + model_time**1.2 / 100) / 1000) / 10000
#testdata += trend
N = 10
#trend = bottleneck.move_median(testdata, N)
print(trend)

for i in range(10):
    t1 = ttime.perf_counter()
    uniform_filter1d(testdata, 100)
    #median_partial_detrender(testdata, 100)
    #bottleneck.move_mean(model_flux, 11)
    #bottleneck.move_median(model_flux, 11)
    t2 = ttime.perf_counter()
    print("Running median detrending", t2-t1)


plt.plot(model_time, model_flux, linewidth=1, color="blue")
#plt.plot(model_time, detrend, linewidth=1, color="red")
plt.scatter(model_time, testdata, s=1, color="black")
plt.show()

polished = testdata - model_flux + 1 #/ detrend


flatten_lc, trend_lc = flatten(model_time, polished, window_length=1, method='biweight', return_trend=True)
plt.plot(model_time, trend_lc, color="red")
plt.scatter(model_time, polished, s=1, color="black")
plt.show()

result = flatten_lc  #polished-trend_lc+1
plt.scatter(model_time, result, s=1, color="black")
#plt.plot(model_time, detrend, color="red")

plt.show()
#testdata = testdata/detrend
#plt.plot(model_time, model_flux, linewidth=1, color="blue")
#plt.scatter(model_time, testdata, s=1, color="black")
#plt.show()


epoch_distance = 365.25
epoch_duration = 2
fig, axs = plt.subplots(2, 3)
axs[0,0].plot(model_time, model_flux, color="black")
axs[0,0].scatter(model_time, testdata, color="black", s=0.5)
axs[0,0].set_xlim(t0_bary - epoch_duration/2, t0_bary + epoch_duration/2)
axs[1,0].scatter(model_time, model_flux-testdata, color="black", s=0.5)
axs[1,0].set_xlim(t0_bary - epoch_duration/2, t0_bary + epoch_duration/2)

axs[0,1].plot(model_time, model_flux, color="black")
axs[0,1].scatter(model_time, testdata, color="black", s=0.5)
axs[0,1].set_xlim(t0_bary - epoch_duration/2 + epoch_distance, t0_bary + epoch_duration/2  + epoch_distance)
axs[1,1].scatter(model_time, model_flux-testdata, color="black", s=0.5)
axs[1,1].set_xlim(t0_bary - epoch_duration/2 + epoch_distance, t0_bary + epoch_duration/2  + epoch_distance)

axs[0,2].plot(model_time, model_flux, color="black")
axs[0,2].scatter(model_time, testdata, color="black", s=0.5)
axs[0,2].set_xlim(t0_bary - epoch_duration/2 + 2 * epoch_distance, t0_bary + epoch_duration/2  + 2 * epoch_distance)
axs[1,2].scatter(model_time, model_flux-testdata, color="black", s=0.5)
axs[1,2].set_xlim(t0_bary - epoch_duration/2 + 2 * epoch_distance, t0_bary + epoch_duration/2  + 2 * epoch_distance)
plt.savefig('planet_moon_model_injected.pdf')

"""
from transitleastsquares import transitleastsquares
model = transitleastsquares(model_time, testdata)
results = model.power(period_min=300)
print('Period', format(results.period, '.5f'), 'd')
print(len(results.transit_times), 'transit times in time series:', \
        ['{0:0.5f}'.format(i) for i in results.transit_times])
print('Transit depth', format(results.depth, '.5f'))
print('Best duration (days)', format(results.duration, '.5f'))
print('Signal detection efficiency (SDE):', results.SDE)
print('SNR:', results.snr)

plt.figure()
ax = plt.gca()
ax.axvline(results.period, alpha=0.4, lw=3)
plt.xlim(np.min(results.periods), np.max(results.periods))
for n in range(2, 10):
    ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
    ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
plt.ylabel(r'SDE')
plt.xlabel('Period (days)')
plt.plot(results.periods, results.power, color='black', lw=0.5)
plt.xlim(0, max(results.periods))
plt.show()

plt.figure()
plt.plot(results.model_folded_phase, results.model_folded_model, color='red')
plt.scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
plt.xlim(0.48, 0.52)
plt.ticklabel_format(u123 seOffset=False)
plt.xlabel('Phase')
plt.ylabel('Relative flux')
plt.show()
"""
# 197sec baseline
# 261sec stuff removed but still has xp yp xm ym
# 216sec rump xp yp
# 214sec with 0 mass ratio
# 113sec after bugfix
# 109sec new baseline
# 103sec xbary loop insourced, 105, 105, 113
# 115sec partial z**2, 120
# 123 sec back to full
# 104 sec back to original, 98
# 105sec with q1q2

# Define upper and lower bounds of Pandora JADE search; in physical units
# Values not provided will be taken from wide defaults (e.g., all inclinations)
# Tighter values increase probability and speed of convergence
# You NEED to give at least per_bary and t0_bary bounds to make a fit possible
# (otherwise, we'd need to run a planet-only search first)
bounds = {
   "per": (1, 400),  # Period [days]
   #"t0":  (0, 1),    # Time of inferior conjunction [days]
   "M_star": (0.5, 1.5)
   #"a":   (2, 1000),      # Semi-major axis [stellar radii]
   #"b":   (0.2, 0.4),
   #"r":   (0.08, 0.1)
    }
"""
bounds_ultranest = {
#"per", "a", "r", "b", "t0", "q1", "q2"
   "per": (365, 365.5),  # Period [days]
   "a":   (150, 270),      # Semi-major axis [stellar radii]
   "r":   (0.08, 0.12),
   "b":   (0, 1),
   "t0":  (11, 12),    # Time of inferior conjunction [days]
   "q1": (0,1),
   "q2": (0,1)
    }
from planejade_caller import fit_ultranest, fit_emcee
from ultranest.plot import cornerplot
"""

"""
result = fit_emcee(model_time, testdata, yerr, bounds_ultranest)
import corner
emcee_plot = corner.corner(
    result.flatchain, 
    labels=result.var_names,
    truths=list(result.params.valuesdict().values())
    )
plt.savefig("cornerplot_emcee.pdf", bbox_inches='tight')
"""
"""
result_planet_only = fit_ultranest(
    time=model_time, 
    flux=testdata, 
    yerr=yerr,
    bounds=bounds_ultranest,
    live_points=1000
    )
cornerplot(result_planet_only)
plt.savefig("cornerplot_ultranest_po.pdf", bbox_inches='tight')
"""
# Run search for planet+moon
print("Run planet-only search")
result_planet_only = fit(
    time=model_time, 
    flux=testdata, 
    yerr=yerr,
    bounds=bounds,
    moon=False,
    live_points=1000,
    batches=1  # 32 appears to be sufficient  
    )
print("Maximum log-likelihood:", round(result_planet_only["logl"], 2))
print("BIC:", round(result_planet_only["BIC"],2))
print("AIC:", round(result_planet_only["AIC"],2))
print("points", result_planet_only["points"])
# truth: -141
#p = 
"""
u1, u2 = ld_convert(result_planet_only["points"][5], result_planet_only["points"][6])
result_planet_only["points"][5] = u1
result_planet_only["points"][6] = u2
"""
# Plot recovered planet-only light curve
recovered_flux = lc(result_planet_only["points"], model_time)

plt.clf()
plt.close()
polished = (testdata - recovered_flux) + 1
flatten_lc, trend_lc = flatten(model_time, polished, window_length=0.3, method='biweight', return_trend=True)
plt.scatter(model_time, flatten_lc, s=1, color="black")
plt.plot(model_time, model_flux, color="blue")
plt.plot(model_time, recovered_flux, color="red")

plt.show()



fig, axs = plt.subplots(2, 3)
axs[0,0].plot(model_time, recovered_flux, color="black")
axs[0,0].plot(model_time, recovered_flux, color="black")
axs[0,0].scatter(model_time, testdata, color="red", s=0.5)
axs[0,0].set_xlim(t0_bary - epoch_duration/2, t0_bary + epoch_duration/2)
axs[1,0].scatter(model_time, recovered_flux-testdata, color="red", s=0.5)
axs[1,0].set_xlim(t0_bary - epoch_duration/2, t0_bary + epoch_duration/2)

axs[0,1].plot(model_time, recovered_flux, color="black")
axs[0,1].scatter(model_time, testdata, color="red", s=0.5)
axs[0,1].set_xlim(t0_bary - epoch_duration/2 + epoch_distance, t0_bary + epoch_duration/2  + epoch_distance)
axs[1,1].scatter(model_time, recovered_flux-testdata, color="red", s=0.5)
axs[1,1].set_xlim(t0_bary - epoch_duration/2 + epoch_distance, t0_bary + epoch_duration/2  + epoch_distance)

axs[0,2].plot(model_time, recovered_flux, color="black")
axs[0,2].scatter(model_time, testdata, color="red", s=0.5)
axs[0,2].set_xlim(t0_bary - epoch_duration/2 + 2 * epoch_distance, t0_bary + epoch_duration/2  + 2 * epoch_distance)
axs[1,2].scatter(model_time, recovered_flux-testdata, color="red", s=0.5)
axs[1,2].set_xlim(t0_bary - epoch_duration/2 + 2 * epoch_distance, t0_bary + epoch_duration/2  + 2 * epoch_distance)
plt.savefig('planet_only_model_recovered.pdf')
