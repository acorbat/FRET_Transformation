# FRET_Transformation

Design to analyze a set of parallel and perpendicular intensity images and produce the corresponding anisotropy curves. Spurious curves are fitted and then manually filtered when necessary. The latter are used to estimate enzyme activity.

## anisotropy_functions

This package contains typical functions used to translate intensity values to and from anisotropy and fluorescence. It has no dependencies except for python 3.x.

## transformation

This package contains typical functions used to treat the anisotropy curves such as sigmoid region selectors, transformation from anisotropy to monomer fraction curves and its parameters optimization.

Needs numpy, scipy.optimize, matplotlib, anisotropy_functions and caspase_fit.

## caspase_fit

This package contains functions required to simulate the simple substrate, enzyme and product model and fit it to the transformed monomer fraction curve. It can also be used to numerically derive monomer fraction curve and numerically estimate the aproximation for enzyme and complex substrate:enzyme curves.

Needs numpy and scipy.

## image_process

This package contains the functions used to correct the crossed intensity images for background and G factor. It latter applies the masks (or erodes and applies masks) to extract the intensity estimators for each object/cell and constructs the corresponding DataFrame.

Needs numpy, pandas, tifffile, scipy.ndimage and skimage.morphology.

## filter_data

This package contains the functions used to filter the curves obtained from the feature extraction algorithm from the images. It calculates anisotropy from the crossed intensities estimated. It applies a windowed fit to the curves to find the sigmoid transitions and then a quick simple filter to discard obvious artifacts. It can be used to filter correct sigmoid transitions and, later on, to select the best describing parameters for those curves from the windowed fit.

Needs numpy, matplotlib, anisotropy_functions, transformation and caspase_fit.