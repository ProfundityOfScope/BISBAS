#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:53:50 2022

@author: bruzewskis
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import tracemalloc
from scipy.io import netcdf_file
import h5py

def imviz(x,y,z1,z2, sig=2, name='dummy.grd'):
    
    vl = np.nanmean(z1) - sig * np.nanstd(z1)
    vh = np.nanmean(z1) + sig * np.nanstd(z1)
    fig = plt.figure(figsize=(16,8), dpi=192)
    ax = plt.subplot2grid((1,2),(0,0))
    plt.pcolormesh(x, y, z1, shading='auto', norm=Normalize(vl, vh),
                   cmap='Spectral')
    plt.colorbar()
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title('Blah')
    
    
    vl = np.nanmean(z2) - sig * np.nanstd(z2)
    vh = np.nanmean(z2) + sig * np.nanstd(z2)
    ax = plt.subplot2grid((1,2),(0,1))
    plt.pcolormesh(x, y, z2, shading='auto', norm=Normalize(vl, vh),
                   cmap='Spectral')
    plt.colorbar()
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title('Blah')
    
    imname = name.replace('grd', 'png')
    plt.tight_layout()
    # plt.savefig(f'/Users/bruzewskis/Dropbox/bisbasgenmap.png', bbox_inches='tight')
    plt.show()

i = 15
with h5py.File('/Users/bruzewskis/Downloads/timeseries.h5', 'r') as f:
    z = f['displacements']
    t = f['t']
    im1 = z[i][:]
    x = f['x'][:]
    y = f['y'][:]
    
    file = f'ts_mm_{t[i]:04d}.grd'
    print(t[i], file)
    
tgt = f'/Users/bruzewskis/Downloads/isbas_ground_truth/timeseries_nodetrend/{file}'
with netcdf_file(tgt, 'r') as f2:
    im2 = f2.variables['z'][:]
    
X,Y = np.meshgrid(x,y)
imviz(X, Y, im2, im1)