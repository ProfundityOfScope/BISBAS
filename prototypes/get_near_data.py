#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:26:03 2022

@author: bruzewskis
"""

from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


tgt = '/Users/bruzewskis/Documents/Projects/bifrost_isbas/isbas/test/intf/'
for r,d,f in os.walk(tgt):
    for file in f:
        if file.endswith('.grd'):
            file = netcdf.netcdf_file(os.path.join(r, file))

            n = 26
            stride = 30
            picks = np.arange(n*stride, (n+1)*stride)
            
            # Name these for easier referencing
            x = file.variables['x']
            y = file.variables['y']
            z = file.variables['z']
            
            # Grab full copies just for testing, delete later
            xd = x[:].copy()
            yd = y[:].copy()
            zd = z[:].copy()
            
            out = (xd,yd,zd)
            name = r.split('/')[-1] + '.p'
            print(name)
            pickle.dump(out, open(name, 'wb'))
            
            file.close()

# # Pick out coordinates we want, wrapping as if meshgridded
# # xt = np.take(file.variables['z'][picks%], picks, mode='wrap')
# xt = x[picks%x.shape[0]].copy()
# yt = y[picks//x.shape[0]].copy()

# # Find correct indices and just pluck out those
# zind = np.unravel_index(picks, z.shape)
# zt = z[zind].copy()

# # Compare
# fig = plt.figure(figsize=(10,5))
# plt.subplot2grid((1,2),(0,0))
# plt.scatter(xt, yt, c=zt, vmin=zd.min(), vmax=zd.max())
# plt.xlim(xd.min()-0.01, xd.max()+0.01)
# plt.ylim(yd.min()-0.01, yd.max()+0.01)

# X,Y = np.meshgrid(xd,yd)
# plt.subplot2grid((1,2),(0,1))
# plt.pcolormesh(X,Y,zd, shading='auto')
# plt.show()

# def get_near_data_moore(x,y,image,x0,y0,numpix=10):
    
    
#     # This is how we would deal with a non-uniform spacing
#     xp = np.round(np.interp(x0, x, np.arange(len(x)))).astype(int)
#     yp = np.round(np.interp(y0, y, np.arange(len(y)))).astype(int)
    
#     # Expand moore until you exceed some threshold, account for nans
    
#     goodpix = 0
#     while goodpix>numpix:
#         break
    
#     mr = 2
#     # Deal with image edges
#     lxp = max([xp-mr,0])
#     lyp = max([yp-mr,0])
    
#     # Extract the subim
#     subim = image[lyp:yp+mr+1, lxp:xp+mr+1]
    
#     # Generate coordinates
#     xs = x[lxp:xp+mr+1]
#     ys = y[lyp:yp+mr+1]
#     XS,YS = np.meshgrid(xs, ys)
#     xn, yn = XS.ravel(), YS.ravel()
    
#     return xn,yn,np.mean(subim),np.median(subim)
    
# def get_near_data(X,Y,image,x0,y0,numpix):
#     # get the mean and median of numpix points near x0,y0, and also return list of
#     # the numpix nearest non-nan X,Y coordinates
#     distarr=np.sqrt((X-x0)**2+(Y-y0)**2)
#     distmask=np.ma.array(distarr,mask=np.isnan(image))
#     nearindx=np.ma.argsort(distmask.ravel())[0:numpix]
#     meannear=np.mean(image.ravel()[nearindx])
#     mediannear=np.median(image.ravel()[nearindx])
#     xn=X.ravel()[nearindx]
#     yn=Y.ravel()[nearindx]
#     return xn,yn,meannear,mediannear

# def ref_igrams(X,Y,data,reflon,reflat,refrad):
#     xn,yn,meannear,mediannear=get_near_data(X,Y,data,reflon,reflat,refrad)
#     data -= mediannear
#     return data

# x0 = 103.696
# y0 = 1.215
# # test = ref_igrams(X, Y, zd, x0, y0, 10)

# xn1,yn1,mean1,median1 = get_near_data(X,Y,zd,x0,y0,10)
# xn2,yn2,mean2,median2 = get_near_data_moore(xd,yd,zd,x0,y0,10)
# print(mean1, median1)
# print(mean2, median2)

# plt.figure(figsize=(6,5), dpi=300)
# plt.pcolormesh(X,Y,zd, shading='nearest')
# plt.colorbar()
# plt.scatter(xn1, yn1, ec='w', fc='none', s=25, marker='d')
# plt.scatter(xn2, yn2, ec='k', fc='none', s=60, marker='s')
# plt.scatter(x0, y0, s=5)
# plt.xlim(103.65,103.75)
# plt.ylim(1.15,1.25)