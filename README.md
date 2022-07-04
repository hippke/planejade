# Plan-e-JADE
Exoplanet transit search with ensemble differential evolution and simultaneous stellar noise detrending.

A classical transit search:
1. Get light curve
2. Remove stellar trends e.g. with [wotan](https://github.com/hippke/wotan)
3. Search for transits e.g. with [TLS](https://github.com/hippke/tls)
4. Fit full transit model and create posterior e.g. with emcee

Perfect sensitivity because Plan-e-JADE fits a full Mandel-Agol 9-parameter transit model to the data, simultaneously with a wotan biweight filter to remove stellar trends. Search is done by iteratively fitting with eJADE, an ensemble differential evolution network.

Why?
Maximum sensitivity because the full combined model is more sensitive than TLS (which is more sensitive than BLS). While TLS uses a realistic transit shape including limb darkening, its transit model is fixed for an entire search. Deviations between the assumed and the true transit shape reduce the sensitivity. Separate detrending lead to compromises: Too short filter lengths destroy part of the transit signal, too long filters do not reduce all stellar variability. The combined approach is the perfect solution. 

Why hasn't everybody been doing that?
Speed. It was computationally prohibitive (link to Petascale computing).
Now possible on your laptop <performance metric here>
H
How's that possible?
- Ultra-fast transit model (from Pandora)
- Ultra-fast combination with biweight wotan (link)
- eJADE
