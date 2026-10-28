<p align="center">
  <img src="docs-site/assets/img/ncplot_wordmark.svg" alt="NCPlot" width="380">
</p>

> [!TIP]
> 📖 **New here?** The NCPlot website — installing, a quickstart, the API reference, and more — lives at **[plymouthmarinelaboratory.github.io/ncplot](https://plymouthmarinelaboratory.github.io/ncplot/)**.

[![pypi](https://img.shields.io/pypi/v/ncplot.svg)](https://pypi.python.org/pypi/ncplot/)
[![Documentation Status](https://readthedocs.org/projects/ncplot/badge/?version=latest)](https://ncplot.readthedocs.io/en/latest/?badge=latest)


# NCPlot

Easy interactive viewing of NetCDF files in Python.

An easy to use Python (3.6 and above) package for quickly plotting the contents of NetCDF files or xarray datasets in Python or on the command line. 

This package is designed to work with [CF](https://cfconventions.org/) compliant NetCDF data.

## Installation

To install from conda:
```sh
conda install -c conda-forge ncplot

```

To install from pypi:
```sh
pip install ncplot 
```

Install the development version using using pip:
```sh
pip install git+https://github.com/pmlmodelling/ncplot.git
```



## How to use in Python



The package is made up of a simple and easy to use function: view. To visualize everything in a file:

```sh
from ncplot import view

view("example.nc")

```

or to visualize a specific variable or list of variables:

```sh
from ncplot import view

view("example.nc", vars)

```

where vars is either a string representing a variable or a list of variables.


If you want to plot xarray datasets dataarrays, first import the xarray accessor, then plot in a similar way to xarray's built in plotting method:


```sh
import ncplot.xarray
import xarray as xr

ds = xr.open_dataset("example.nc") 
ds.ncplot.view()

```




## How to use on the command line

Command line usage is as simple as:


```sh


ncplot example.nc

```












# Reference and tutorials

A full API reference, and a how-to guide are available at the [NCPlot website](https://plymouthmarinelaboratory.github.io/ncplot/) and on [readthedocs](https://ncplot.readthedocs.io/en/latest/).








