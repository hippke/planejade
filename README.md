Plan-e-JADE
====================

Optimal sensitivity for your exoplanet transit search with simultaneous stellar noise detrending.

>Fit a full Mandel-Agol transit model to the data & simultaneously a wōtan biweight filter to remove stellar trends. Search by iterative fitting with an ensemble differential evolution network. Provides posterior diagnostics and cornerplots.

Install with ``pip install planejade``

Minimal example:
```
from planejade import search
import corner
import matplotlib.pyplot as plt
result = search(time, flux, yerr, bounds=None)
print("Evidence:", result.diagnostics)
cornerplot = corner.corner(result.flatchain, labels=result.var_names)
plt.plot(time, result.model)
```

The combined model is even more sensitive than [TLS](https://github.com/hippke/tls) (which is more sensitive than BLS). While TLS uses a realistic transit shape including limb darkening, its transit model is fixed for an entire search. Deviations between the assumed and the true transit shape reduce the sensitivity. Separate detrending lead to compromises: Too short filter lengths destroy part of the transit signal, too long filters do not reduce all stellar variability. The combined approach is the perfect solution. 

For reference, in a classical transit search the procedure is to:
1. Remove stellar trends e.g. with [wotan](https://github.com/hippke/wotan)
2. Search for transits e.g. with [TLS](https://github.com/hippke/tls)
3. Fit full transit model and create posterior e.g. with emcee

*Plan-e-JADE combines all of these.*

### Q&A
This should already be best-practice!?
- In the past, you needed a [supercomputer](https://ui.adsabs.harvard.edu/abs/2020AJ....159..283T/abstract) for a full search. Now it's possible on your laptop computer.

What makes Plan-e-JADE so much faster?
- Ultra-fast transit model from [Pandora](https://github.com/hippke/pandora) with $7\times10^8$ data points per second on an AMD Ryzen 5950X
- Ultra-fast detrending with the [biweight](https://github.com/hippke/wotan/blob/master/tutorials/02%20Sliders.ipynb) filter from [wotan](https://github.com/hippke/wotan)
- Complete light curve search with ensemble differential evolution (e-JADE), requiring $10^7$ models for convergence. Find planets in Kepler and TESS in a few minutes with optimal sensitivity.

Code: Open source (GPL3), pure Python (compiled with numba and Cython)



