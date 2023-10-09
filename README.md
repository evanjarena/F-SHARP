# F-SHARP

| `F-SHARP`:  | A tool for measuring weak lensing correlations |
|-------------|------------------------------------------------|
| Author:     | Evan J. Arena                                  |

`F-SHARP` (Flexion and SHear Arbitrary Point correlations) is a Python tool for computing both theoretical N-point weak lensing correlations and for measuring the real-space N-point correlations from a dataset.  This version currently handles cosmic flexion and cosmic shear-flexion two-point correlations.  

* (c) Evan J. Arena (Drexel University Department of Physics), 2022.
* For questions, please email `evan.james.arena@drexel.edu.`

## Required Packages
* numpy
* astropy
* scipy
* matplotlib
* pickle
* pandas
* numdifftools
* fastdist
* classy (Python wrapper of the Einstein-Boltzmann code `CLASS` (https://lesgourg.github.io/class_public/class.html)

## Modules

* `coflex_power.py`: Computes theoretical cosmic weak lensing power spectra
* `coflex_twopoint.py`: Computes theoretical cosmic weak lensing two-point correlation functions
* `measure_coflex.py`: Measures cosmic weak lensing two-point correlation functions from a dataset 


