#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:53:50 2022

@author: bruzewskis
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import tracemalloc
from scipy.io import netcdf_file
import h5py

def imviz(x,y,z1,z2, sig=3, name=''):

    vl = np.nanmean(z1) - sig * np.nanstd(z1)
    vh = np.nanmean(z1) + sig * np.nanstd(z1)
    plt.figure(figsize=(16, 14), dpi=192)
    plt.subplot2grid((2, 2), (0, 0))
    plt.pcolormesh(x, y, z1, shading='auto', norm=Normalize(vl, vh),
                   cmap='Spectral')
    plt.colorbar()
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title(name+' his')

    plt.subplot2grid((2, 2), (0, 1))
    plt.pcolormesh(x, y, z2, shading='auto', norm=Normalize(vl, vh),
                   cmap='Spectral')
    plt.colorbar()
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title(name+' mine')

    diff = z1 - z2
    plt.subplot2grid((2, 2), (1, 0))
    vl = np.nanmean(diff) - sig * np.nanstd(diff)
    vh = np.nanmean(diff) + sig * np.nanstd(diff)
    plt.pcolormesh(x, y, diff, shading='auto', norm=Normalize(vl, vh),
                   cmap='Spectral')
    plt.colorbar()
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title(name+' difference')

    plt.subplot2grid((2, 2), (1, 1))
    bb = np.linspace(vl, vh, 25)
    plt.hist(diff.ravel(), bins=bb)
    plt.semilogy()

    plt.tight_layout()
    pngname = name.replace('.grd', '.png')
    plt.savefig(f'/Users/bruzewskis/Dropbox/comparisons/{pngname}.png',
                bbox_inches='tight')
    plt.show()


for i in range(20):

    with h5py.File('/Users/bruzewskis/Downloads/timeseries.h5', 'r') as f:
        z = f['displacements']
        t = f['t']
        im1 = z[i][:] * -18.303
        x = f['x'][:]
        y = f['y'][:]

        file = f'ts_mm_{t[i]:04d}.grd'
        print(t[i], file)
  
    tgt = f'/Users/bruzewskis/Downloads/isbas_ground_truth/timeseries_nodetrend/{file}'
    with netcdf_file(tgt, 'r') as f2:
        im2 = f2.variables['z'][:]

    X, Y = np.meshgrid(x, y)
    imviz(X, Y, im2, im1, name=file)
