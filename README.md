![Logo](https://github.com/hippke/planejade/blob/main/logo_planejade.png?raw=true)
====================

## Optimal sensitivity for your exoplanet transit search with simultaneous stellar noise detrending.

[![pip](https://img.shields.io/badge/pip-install%20planejade-blue.svg)](https://pypi.org/project/planejade/)
[![Documentation](https://img.shields.io/badge/documentation-%E2%9C%93-blue.svg)](https://planejade.readthedocs.io/en/latest/index.html)
[![Image](https://img.shields.io/badge/Examples-%E2%9C%93-blue.svg)](https://github.com/hippke/planejade/tree/main/examples)
[![Image](https://img.shields.io/badge/arXiv-2205.09410-blue.svg)](https://arxiv.org/abs/2205.09410)

>Fit a full Mandel-Agol transit model to the data & simultaneously a wōtan biweight filter to remove stellar trends. Search by iterative fitting with an ensemble differential evolution network. Obtain posterior diagnostics and cornerplots.

### Minimal example for blind search, cornerplot, diagnostics, and posterior model:
```
from planejade import search
import corner
import matplotlib.pyplot as plt
result = search(time, flux, yerr, bounds=None)
print(result.diagnostics)
cornerplot = corner.corner(result.chain, labels=result.var_names)
plt.plot(time, result.model)
plt.scatter(time, result.flux)
```

The combined model is even more sensitive than [TLS](https://github.com/hippke/tls) (which is more sensitive than BLS). While TLS uses a realistic transit shape including limb darkening, its transit model is fixed for an entire search. Deviations between the assumed and the true transit shape reduce the sensitivity. Separate detrending lead to compromises: Too short filter lengths destroy part of the transit signal, too long filters do not remove all stellar variability. The combined approach is the perfect solution. 

For reference, in a classical transit search the procedure is to:
1. Remove stellar trends e.g. with [wōtan](https://github.com/hippke/wotan)
2. Search for transits e.g. with [TLS](https://github.com/hippke/tls)
3. Fit full transit model and create posterior e.g. with emcee

*Plan-e-JADE combines all of these.*

### Q&A
This should already be best-practice!?
- In the past, you needed a [supercomputer](https://ui.adsabs.harvard.edu/abs/2020AJ....159..283T/abstract) for a full search. Now it's possible on your laptop computer.

What makes Plan-e-JADE so much faster?
- Ultra-fast transit model from [Pandora](https://github.com/hippke/pandora) with $7\times10^8$ data points per second on an AMD Ryzen 5950X
- Ultra-fast detrending with the [biweight](https://github.com/hippke/wotan/blob/master/tutorials/02%20Sliders.ipynb) filter from [wōtan](https://github.com/hippke/wotan)
- Complete light curve search with ensemble differential evolution (e-JADE), requiring $\mathcal{O}(10^7)$ model evaluations for convergence. Find planets in Kepler and TESS in a few minutes with optimal sensitivity.

### Code
Pure Python, compiled with numba and Cython for maximum speed
Open source license: GPLv3


### Attribution
Please cite Hippke et al. (2022) if you find this code useful in your research. The BibTeX entry for the paper is:
```
@article{
}
```
