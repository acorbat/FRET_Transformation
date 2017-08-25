# FRET_Transformation

Design to analyze a set of parallel and perpendicular intensity images and produce the corresponding anisotropy curves. Spurious curves are fitted and then manually filtered when necessary. The latter are used to estimate enzyme activity.

## anisotropy_functions

This package contains typical functions used to translate intensity values to and from anisotropy and fluorescence. It has no dependencies except for python 3.x.

## transformation

This package contains typical functions used to treat the anisotropy curves such as sigmoid region selectors, transformation from anisotropy to monomer fraction curves and its parameters optimization.

Needs numpy, scipy.optimize, matplotlib, anisotropy_functions and caspase_fit.

