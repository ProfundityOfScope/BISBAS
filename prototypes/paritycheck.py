#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:48:37 2022

@author: bruzewskis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:56:00 2022

@author: bruzewskis
"""

from scipy.io import netcdf
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import pickle
from tqdm import trange,tqdm
from time import time
import h5py

def imviz(x, y, z, sigma=3):
    """Use this functio to do something."""

    low = np.nanmean(z) - sigma * np.nanstd(z)
    high = np.nanmean(z) + sigma * np.nanstd(z)
    norm = Normalize(low, high)
    
    plt.figure(figsize=(10, 11), dpi=192)
    plt.pcolormesh(x, y, z, shading='auto', norm=norm, cmap='Spectral')
    plt.colorbar(location='bottom', pad=0.05, label='Displacement [mm]')
    
    plt.xlim(np.min(x), np.max(x))
    plt.xlabel('Longitude [deg]')
    
    plt.ylim(np.min(y), np.max(y))
    plt.ylabel('Latitude [deg]')
    
    plt.title('Mine')
    plt.grid()
    plt.tight_layout()
    plt.show()

with h5py.File('/Users/bruzewskis/Downloads/timeseries.h5', 'r') as fo:
    
    i = 10
    imviz(fo['x'], fo['y'], fo['displacements'][:,:,i])
    pass
