#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:01:40 2022

@author: bruzewskis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy import ndimage
from scipy.io import netcdf
import sys

tgt = '/Users/bruzewskis/Documents/Projects/BISBAS/testing/intf'
for r,d,f in os.walk(tgt):
    for file in f:
        path = os.path.join(r,file)
        
        print(path)
        
        data = netcdf.netcdf_file(path, mmap=True)
        xd = data.variables['lon'][:]
        yd = data.variables['lat'][:]
        zd = data.variables['z'][:]
        # data.close()
        
        # Plot original data
        zm = zd - np.nanmean(zd)
        plt.pcolormesh(xd, yd, zm, norm=SymLogNorm(1e-4), shading='auto')
        plt.colorbar()
        
        # 
        
        sys.exit()