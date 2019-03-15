# fSIM with speckle in Python3

This is a Python3 implementation of fluorescence structured illumination microscopy algorithm with speckle <br/>
```fSIM_speckle_simulation.ipynb```: Jupyter notebook for creating simulation data for fSIM processing <br/>
```Preprocess.ipynb```: Jupyter notebook for generating appropriate exp dataset for optimization <br/>
```fSIM_main.ipynb```: Jupyter notebook for main optimization algorithm to process dataset <br/>
```fSIM_func.py```: Processing functions with numpy implementation <br/>
```fSIM_func_af.py```: Processing functions with arrayfire implementation <br/>
```dftregistration.py```: DFT registration python code translated by [1] to python from original MATLAB code in [2] <br/>


[1] https://github.com/keflavich/image_registration <br/>
[2] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel image registration algorithms," Opt. Lett. 33, 156-158 (2008). <br/>

## Environment requirement
Python 3.6, ArrayFire <br/>

1. Follow http://arrayfire.org/docs/index.htm for ArrayFire installation
2. Follow https://github.com/arrayfire/arrayfire-python to install ArrayFire Python and set the path to the libraries

## Data download
You can find sample experiment data from here: TBD
