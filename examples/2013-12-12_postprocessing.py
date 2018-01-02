import pathlib
import pandas as pd
import matplotlib.pyplot as plt

from fret_transformation import filter_data as fd
from fret_transformation import time_study as ts

data_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/')
data_dir = data_dir.joinpath('2017-10-16_complex_noErode_order05.pandas')
data = pd.read_pickle(str(data_dir))

Differences_tags = ['TFP_to_YFP', 'TFP_to_mKate', 'YFP_to_mKate']
data = ts.add_differences(data, Difference_tags=Differences_tags)

data = fd.filter_derived(data)

save_dir = data_dir.parent
save_dir = save_dir.joinpath(data_dir.stem+'_filtered_derived.pandas')
data.to_pickle(save_dir)
