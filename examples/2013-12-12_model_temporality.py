# import packages to be used
import sys
import pathlib
import pandas as pd
import lmfit as lm

# add path to local library
sys.path.insert(0,
            '/mnt/data/Laboratorio/Imaging three sensors/FRET_Transformation')
from fret_transformation import caspase_model as cm

#%% Load data
work_dir = pathlib.Path(
            '/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images')
data_dir = work_dir.joinpath('2017-10-16_complex_noErode_order05.pandas')
df = pd.read_pickle(str(data_dir))

#%% 
