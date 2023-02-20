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

from scipy.io import netcdf_file
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import pickle
from tqdm import trange,tqdm
from time import time
import h5py

def imviz(x, y, z1, z2, sigma=3):
    """Use this functio to do something."""

    low = np.nanmean(z1) - sigma * np.nanstd(z1)
    high = np.nanmean(z1) + sigma * np.nanstd(z1)
    norm = Normalize(low, high)
    
    plt.figure(figsize=(10, 9), dpi=192)
    plt.subplot2grid((3,4),(0,0), rowspan=2, colspan=2)
    plt.pcolormesh(x, y, z1, shading='auto', norm=norm, cmap='Spectral')
    plt.colorbar(location='bottom', pad=0.05)
    
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.xticks([])
    plt.yticks([])
    
    plt.title('Original')
    
    plt.subplot2grid((3,4),(0,2), rowspan=2, colspan=2)
    plt.pcolormesh(x, y, z2, shading='auto', norm=norm, cmap='Spectral')
    plt.colorbar(location='bottom', pad=0.05)
    
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.xticks([])
    plt.yticks([])
    
    plt.title('Bifrost')
    
    diff = (z1-z2).ravel()
    print(np.nanmean(diff))
    plt.subplot2grid((3,4),(2,0), colspan=4)
    plt.hist(diff, bins=np.linspace(-1000, 1000))
    plt.semilogy()
    
    plt.tight_layout()
    plt.show()

with h5py.File('/Users/bruzewskis/Downloads/timeseries.h5', 'r') as fo:
    
    i = 1
    date = fo['t'][i]
    
    real = f'/Users/bruzewskis/Downloads/isbas_ground_truth/timeseries_nodetrend/ts_mm_{date:04d}.grd'
    with netcdf_file(real) as fr:
        im_true = fr.variables['z'][:]
        
    imviz(fo['x'], fo['y'],im_true, fo['displacements'][:,:,i])
