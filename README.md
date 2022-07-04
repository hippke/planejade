# Plan-e-JADE
Exoplanet transit search with ensemble differential evolution and simultaneous stellar noise detrending.

A classical transit search:
1. Get light curve
2. Remove stellar trends e.g. with [wotan](https://github.com/hippke/wotan)
3. Search for transits e.g. with [TLS](https://github.com/hippke/tls)
4. Fit full transit model and create posterior e.g. with emcee

With Plan-e-JADE, a full Mandel-Agol 9-parameter transit model is fit to the data, simultaneously with a wotan biweight filter to remove stellar trends. This combined transit+noise model is iteratively fit to search for transits using eJADE, an ensemble differential evolution network.
