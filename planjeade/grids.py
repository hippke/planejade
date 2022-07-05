import numpy as np
from numpy import sqrt, pi, cos, arcsin, empty
from numba import jit


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def timegrid(t0_bary, epochs, epoch_duration, cadences_per_day, epoch_distance, supersampling_factor):
    # epoch_distance is the fixed constant distance between subsequent data epochs
    # Should be identical to the initial guess of the planetary period
    # The planetary period `per_bary`, however, is a free parameter
    ti_epoch_midtimes = np.arange(
        start=t0_bary,
        stop=t0_bary + epoch_distance * epochs,
        step=epoch_distance,
    )

    # arrays of epoch start and end dates [day]
    t_starts = ti_epoch_midtimes - epoch_duration / 2
    t_ends = ti_epoch_midtimes + epoch_duration / 2

    # "Morphological light-curve distortions due to finite integration time"
    # https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1758K/abstract
    # Data gets smeared over long integration. Relevant for e.g., 30min cadences
    # To counter the effect: Set supersampling_factor = 5 (recommended value)
    # Then, 5x denser in time sampling, and averaging after, approximates effect
    if supersampling_factor < 1:
        print("supersampling_factor must be positive integer")
    supersampled_cadences_per_day = cadences_per_day * int(supersampling_factor)
    supersampled_cadences_per_day = cadences_per_day * int(supersampling_factor)

    cadences = int(supersampled_cadences_per_day * epoch_duration)
    time = np.empty(shape=(epochs, cadences))
    for epoch in range(epochs):
        time[epoch] = np.linspace(t_starts[epoch], t_ends[epoch], cadences)
    return time.ravel()

