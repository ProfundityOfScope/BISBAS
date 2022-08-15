#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a prototype implementation of a helper function to extract data near
a specific point

Created on Thu Jun 30 13:26:03 2022
@author: bruzewskis
"""

from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import os

def get_near_data_moore(x,y,image,x0,y0,numpix=20, max_moore=5, ax=None):
    
    # This is how we would deal with a non-uniform spacing
    xp = np.interp(x0, x, np.arange(len(x)))
    yp = np.interp(y0, y, np.arange(len(y)))
    
    min_size = np.ceil(np.sqrt(numpix)).astype(int)
    for moore_size in np.arange(min_size,10):
        moore_rad = moore_size/2
        
        if not ax is None:
            pxscale = x[1] - x[0]
            r = moore_rad * pxscale
            size = moore_size * pxscale
            ax.add_artist(Rectangle((x0-r,y0-r),size,size, fc='none', ec='k', ls='--', lw=1) )
        
        xmin = np.ceil( xp - moore_rad ).astype(int)
        ymin = np.ceil( yp - moore_rad ).astype(int)
        xmax = xmin + moore_size
        ymax = ymin + moore_size
        
        subim = image[ymin:ymax, xmin:xmax]
        if np.sum(~np.isnan(subim))>=numpix:
            #return slice(ymin, ymax), slice(xmin, xmax)
            break
    else:
        print('Too big, discard this point')
    
    # Generate coordinates
    xs = x[xmin:xmax]
    ys = y[ymin:ymax]
    XS,YS = np.meshgrid(xs, ys)
    xn, yn = XS.ravel(), YS.ravel()
    return xn,yn,np.nanmean(subim),np.nanmedian(subim)
    
def get_near_data(X,Y,image,x0,y0,numpix):
    # get the mean and median of numpix points near x0,y0, and also return list of
    # the numpix nearest non-nan X,Y coordinates
    distarr=np.sqrt((X-x0)**2+(Y-y0)**2)
    distmask=np.ma.array(distarr,mask=np.isnan(image))
    nearindx=np.ma.argsort(distmask.ravel())[0:numpix]
    meannear=np.mean(image.ravel()[nearindx])
    mediannear=np.median(image.ravel()[nearindx])
    xn=X.ravel()[nearindx]
    yn=Y.ravel()[nearindx]
    return xn,yn,meannear,mediannear

def ref_igrams(X,Y,data,reflon,reflat,refrad):
    xn,yn,meannear,mediannear=get_near_data(X,Y,data,reflon,reflat,refrad)
    data -= mediannear
    return data

tgt = '/Users/bruzewskis/Documents/Projects/BISBAS/prototypes/pickled'
x = pickle.load(open(f'{tgt}/xvec.p', 'rb'))
y = pickle.load(open(f'{tgt}/yvec.p', 'rb'))
X,Y = np.meshgrid(x,y)
z = pickle.load(open(f'{tgt}/datamatrix.p', 'rb'))[3]
z[20,19] = np.nan

x0 = 103.6901 + 0.009 * 0.9
y0 = 1.210 + 0.009 * 0.3

fig = plt.figure(figsize=(6,5), dpi=300)
ax = fig.add_subplot()
plt.pcolormesh(X,Y,z, shading='nearest')
plt.colorbar()

n = 9
xn1,yn1,mean1,median1 = get_near_data(X,Y,z,x0,y0, n)
xn2,yn2,mean2,median2 = get_near_data_moore(x,y,z,x0,y0, n, ax=ax)
print(mean1, median1)
print(mean2, median2)

plt.scatter(xn1, yn1, ec='w', fc='none', s=60, marker='s', label='Eric')
plt.scatter(xn2, yn2, ec='w', fc='none', s=200, marker='o', label='Seth')
plt.legend()
plt.scatter(x0, y0, s=50, ec='k', fc='w')
plt.xlim(x0-0.05, x0+0.05)
plt.ylim(y0-0.05, y0+0.05)
plt.title(f'Requested {n} points around reference')