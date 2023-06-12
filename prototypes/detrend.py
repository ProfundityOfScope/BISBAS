#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:56:00 2022

@author: bruzewskis
"""

import os
from scipy.io import netcdf_file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.patches import Rectangle
import h5py

from time import time


def detrend_constraints(x,y,image,gpsdat,trendparams=3):
    # for each date, remove a trend by fitting to the entire image,
    # but using a small number of GPS as constraints
    #
    
    X,Y = np.meshgrid(x,y)
    onemat=np.ones(np.shape(X))
        
    #determine which method to use
    if(trendparams == range(np.size(gpsdat,0))):
        invmethod='inv'
    else:
        invmethod='pinv'
    
    #estimate trends for each time series component
    #extract data around the GPS points for use in constraint equations
    C, z = construct_gpsconstraint(0,image, gpsdat,X, Y)

    # Construct image constraints
    G, d = construct_imconstraint(image, x, y)

    # select the specified number of trend parameters
    G = G[:, :trendparams]
    C = C[:, :trendparams]

    # invert for a trend subject to constraint that it matches the value around given GPS points
    K,D,m = constrained_lsq(G, d, C, z, method=invmethod)

    # reconstruct the model
    # model = reconstruct_model_nparams(trendparams, m, X, Y)

    return K,D

def construct_imconstraint(image, x, y):

    X, Y = np.meshgrid(x, y)
    onemat = np.ones(np.shape(X))

    d = image.ravel()
    indx = ~np.isnan(d)
    xG = X.ravel()[indx]
    yG = Y.ravel()[indx]
    onevec = onemat.ravel()[indx]
    d = d[indx]
    G = np.column_stack([onevec, xG, yG, xG**2, yG**2, xG*yG])

    return G, d


def construct_imconstraint2(image, x, y):

    yind, xind = np.where(~np.isnan(image))
    
    xG = x[xind]
    yG = y[yind]
    oG = np.ones_like(xG, dtype=np.float64)
    d = image[yind, xind]
    G = np.column_stack([oG, xG, yG, xG**2, yG**2, xG*yG])

    return G, d


def construct_gpsconstraint(imagenum,image,gpsdat,X,Y):
    # create the matrices used to fit a trend to values near the gps points
    #
    for j in range(np.size(gpsdat,0)):
        x0=gpsdat[j,0]
        y0=gpsdat[j,1]
        numpix=int(gpsdat[j,2])
        if (np.size(gpsdat,axis=1) == 4):
            v0=gpsdat[j,3] #use average velocity
        else:
            v0=gpsdat[j,3+imagenum] #use displacement specified at a particular epoch
        
             
        xn,yn,meannear,mediannear=get_near_data(X,Y,image,x0,y0,numpix)
        
        #constraint equation is that the sum of trend values at all the selected points is equal to the sum of their mean
        #assumes 6 parameters here (this is the maximum for now), remove extra parameters later before fitting
        Gpoint=np.array([numpix, np.sum(xn), np.sum(yn), np.sum(xn**2), np.sum(yn**2), np.sum(xn*yn)])
        dpoint=numpix*(meannear-v0)
        if (j==0):
            G=np.array(Gpoint,ndmin=2)
            d=dpoint
        else:
            G=np.row_stack(( G,Gpoint ))
            d=np.row_stack(( d,dpoint )) 
            
    return G, d


def get_near_data(X,Y,image,x0,y0,numpix):
    # get the mean and median of numpix points near x0,y0, and also return list
    # the numpix nearest non-nan X,Y coordinates
    distarr = np.sqrt((X-x0)**2+(Y-y0)**2)
    distmask = np.ma.array(distarr, mask=np.isnan(image))
    nearindx = np.ma.argsort(distmask.ravel())[0:numpix]
    meannear = np.mean(image.ravel()[nearindx])
    mediannear = np.median(image.ravel()[nearindx])
    xn = X.ravel()[nearindx]
    yn = Y.ravel()[nearindx]
    print(meannear)
    return xn, yn, meannear, mediannear


def constrained_lsq(G,d,C,z=0,method='inv'):

    # double check the shape of d
    d = np.array(d, ndmin=2).T
    if np.size(d, 0) == 1:
        d = d.T
    # double check the shape of C - avoids problem if only one constraint
    C = np.array(C, ndmin=2)

    # form the big matrix -- sorry, these double-parentheses are hard to read
    K = np.column_stack((np.row_stack((2*np.dot(G.T, G), C)),
                         np.row_stack((C.T, np.zeros((np.size(C, 0),
                                                      np.size(C, 0)))))))

    # form the data column
    if np.size(z) != np.size(C, 0):
        z = np.zeros((np.size(C, 0), 1))
    D = np.row_stack((2*np.dot(G.T, d), z))

    # compute the model vector
    if method == 'inv':
        m = np.dot(np.linalg.inv(K), D)
    elif method == 'pinv':
        m = np.dot(np.linalg.pinv(K), D)
    # separate the model from lagrange multipliers
    mm = m[0:np.size(G, 1)]
    return K, D, mm


def data_near(data, x0, y0, min_points=10, max_size=20):
    
    # We don't need to check smaller chunks
    min_size = np.ceil(np.sqrt(min_points)).astype(int)

    # We need to find a good chunk size
    for chunk_size in np.arange(min_size, max_size):

        # Find corners
        xmin = np.ceil(x0 - chunk_size/2).astype(int)
        ymin = np.ceil(y0 - chunk_size/2).astype(int)
        xmax = xmin + chunk_size
        ymax = ymin + chunk_size

        # Grab that bit of the images
        zarr = data[:, ymin:ymax, xmin:xmax]

        # Check if what we grabbed is nice enough
        good_count = np.sum(~np.isnan(zarr), axis=(1, 2))
        if np.all(good_count >= min_points):
            # Skip lugging around the meshgrid
            ym, xm = np.mgrid[ymin:ymax, xmin:xmax]
            break
    else:
        raise ValueError('Couldn\'t find a good chunk, try a different reference')
        
    xm = np.broadcast_to(xm, zarr.shape)
    ym = np.broadcast_to(ym, zarr.shape)
    print(np.nanmean(zarr, axis=(1,2))[10])
    return xm, ym, zarr

def gpsconmine(data, gpsdat):
    
    ng = len(gpsdat)
    nd = len(data)
    Gg = np.zeros((ng, 3, nd))
    dg = np.zeros((ng, nd))
    for i in range(len(gpsdat)):
            # Find a good chunk of data
            xa, ya, za = data_near(data, *gpsdat[i,:3])
            isgood = ~np.isnan(za)
            numgood = np.sum(isgood, axis=(1, 2))
            scale = gpsdat[i,2] / numgood

            # Record it's bulk properties
            Gg[i] = np.row_stack([numgood * scale,
                                  np.sum(xa,  where=isgood, axis=(1,2)) * scale,
                                  np.sum(ya,  where=isgood, axis=(1,2)) * scale])
            dg[i] = (np.nanmean(za, axis=(1, 2)) - gpsdat[i,3]) * numgood * scale
            
    return Gg, dg
        
def myKD(fo, gpsdat):
    
    
    
    d = fo['z'][:].reshape((fo['z'].shape[0], -1)).T
    gooddata = ~np.isnan(d)
    X, Y = np.meshgrid(np.arange(fo['x'].size), np.arange(fo['y'].size))
    G = np.array([np.ones(X.size), X.ravel(), Y.ravel()]).T
    GTG = np.einsum('ij,jk,jl->ikl', G.T, G, gooddata)
    GTd = np.nansum(np.einsum('ij,jk->ijk', G.T, d), axis=1)
    
    Cm, zm = gpsconmine(fo['z'], gpsdat2)
    
    nt = 3
    ng = len(gpsdat)
    nd = np.size(fo['z'], 0)
    K = np.zeros((nt+ng, nt+ng, nd))
    K[:nt, :nt] = 2 * GTG[:nt, :nt]
    K[:nt, nt:] = np.transpose(Cm[:,:nt], (1,0,2))
    K[nt:, :nt] = Cm[:,:nt]

    # Assemble D matrix
    D = np.zeros((ng+nt, nd))
    D[:nt] = 2 * GTd[:nt]
    D[nt:] = zm
    
    
    return K, D


if __name__=='__main__':
    reflat = 36.645836
    reflon = 255.30833
    refnum = 20
    
    # tgt = '/Users/bruzewskis/Downloads/isbas_ground_truth/timeseries_nodetrend/'
    
    # fo = h5py.File(os.path.join(tgt, 'nodetrend.h5'), 'r')
    # refx = np.interp(reflon, fo['x'], np.arange(fo['x'].size))
    # refy = np.interp(reflat, fo['y'], np.arange(fo['y'].size))
    # gpsdat1 = np.array([[reflon, reflat, refnum, 0]])
    # gpsdat2 = np.array([[refx, refy, refnum, 0]])
    
    # factors = np.array([300,600,1000,1800])
    # times = np.array([1.5,3,5,9])
    
    # total_times = 3500*3900/factors * times / 3600
    
    # plt.scatter(factors, total_times)
    
    from scipy.interpolate import make_interp_spline
    
    x = np.sort(np.random.uniform(-2, 2, 10))
    y = x[:,None] * np.random.uniform(-1, 1, 5)
    y += np.random.normal(0, 0.05, y.shape)
    
    for i in range(5):
        plt.scatter(x,y[:,i])
    
    xi = np.linspace(np.min(x), np.max(x), 100)
    t = make_interp_spline(x, y, 2)
    yi = t(xi)
    plt.plot(xi, yi)
    # plt.ylim(-15, 50)
    
    
    # Kh, Dh = detrend_constraints(fo['x'], fo['y'], fo['z'][10], gpsdat1)
    # m1 = np.squeeze(np.linalg.solve(Kh, Dh))[:3]
    # print(m1)
    
    # Km, Dm = myKD(fo, gpsdat2)
    # m2 = np.linalg.solve(Km[:,:,10], Dm[:,10])[:3]
    # print(m2)
    
    # view = (45, 225)
    # fig = plt.figure(figsize=(10,10))
    # ax1 = plt.subplot2grid((2, 2), (0, 0), projection='3d')
    # ax1.view_init(*view)
    
    # x1 = np.linspace(np.min(fo['x']), np.max(fo['x']), 10)
    # y1 = np.linspace(np.min(fo['y']), np.max(fo['y']), 11)
    # X1,Y1 = np.meshgrid(x1,y1)
    # A1 = np.array([np.ones_like(X1), X1, Y1])
    # Z1 = np.dot(np.transpose(A1, (1,2,0)), m1)
    # ax1.plot_surface(X1,Y1,Z1)
    
    # ax2 = plt.subplot2grid((2, 2), (0, 1), projection='3d')
    # ax2.view_init(*view)
    
    # x2 = np.linspace(0, fo['x'].size, 10)
    # y2 = np.linspace(0, fo['y'].size, 11)
    # X2,Y2 = np.meshgrid(x2,y2)
    # A2 = np.array([np.ones_like(X2), X2, Y2])
    # Z2 = np.dot(np.transpose(A2, (1,2,0)), m2)
    # ax2.plot_surface(X2,Y2,Z2)
    
    
    
    # ax3 = plt.subplot2grid((2, 2), (1, 0), projection='3d')
    # ax3.view_init(*view)
    # ax3.plot_surface(X1, Y1, Z2-Z1)