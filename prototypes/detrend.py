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
import tracemalloc


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
    C1, z1 = construct_gpsconstraint(0,image, gpsdat,X, Y)
    C2, z2 = construct_gpsconstraint2(image, gpsdat, x, y)

    # Construct image constraints
    G1, d1 = construct_imconstraint(image, x, y)
    G2, d2 = construct_imconstraint2(image, x, y)  

    # select the specified number of trend parameters
    G1 = G1[:, :trendparams]
    G2 = G2[:, :trendparams]
    C1 = C1[:, :trendparams]
    C2 = C2[:, :trendparams]
    
    print(np.dot(G2.T, G2))
    print(np.dot(G2.T, d2))
    print(d2)

    # invert for a trend subject to constraint that it matches the value around given GPS points
    m1 = constrained_lsq(G1, d1, C1, z1, method=invmethod)
    m2 = constrained_lsq2(G2, d2, C2, z2, invmethod)

    # reconstruct the model
    model1 = reconstruct_model_nparams(trendparams, m1, X, Y)     
    model2 = reconstruct_model_nparams2(m2, x, y)

    return image-model2

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


def construct_gpsconstraint2(image, gpsdat, x, y):
    # create the matrices used to fit a trend to values near the gps points
    #
    for j in range(np.size(gpsdat, 0)):
        x0 = gpsdat[j, 0]
        y0 = gpsdat[j, 1]
        numpix = int(gpsdat[j, 2])
        if np.size(gpsdat, axis=1) == 4:
            v0 = gpsdat[j, 3]  # use average velocity
        else:
            # TODO: FIX THIS
            v0 = gpsdat[j, 3]  # use displacement at a particular epoch
        
             
        xn,yn,meannear,mediannear=get_near_data2(x,y,image,x0,y0,numpix)
        
        #constraint equation is that the sum of trend values at all the selected points is equal to the sum of their mean
        #assumes 6 parameters here (this is the maximum for now), remove extra parameters later before fitting
        Gpoint=np.array([len(xn), np.sum(xn), np.sum(yn), np.sum(xn**2), np.sum(yn**2), np.sum(xn*yn)])
        dpoint=len(xn)*(meannear-v0)
        if (j==0):
            G=np.array(Gpoint,ndmin=2)
            d=dpoint
        else:
            G=np.row_stack(( G,Gpoint ))
            d=np.row_stack(( d,dpoint )) 
            
    return G,d

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

def get_near_data2(x, y, image, x0, y0, moore_size):
    
    # This is how we would deal with a non-uniform spacing
    xp = np.interp(x0, x, np.arange(len(x)))
    yp = np.interp(y0, y, np.arange(len(y)))

    moore_rad = moore_size/2
    xmin = np.ceil(xp - moore_rad).astype(int)
    ymin = np.ceil(yp - moore_rad).astype(int)
    xmax = xmin + moore_size
    ymax = ymin + moore_size

    subim = image[ymin:ymax, xmin:xmax]

    # Generate coordinates
    xs = x[xmin:xmax]
    ys = y[ymin:ymax]
    XS, YS = np.meshgrid(xs, ys)

    good = ~np.isnan(subim.ravel())
    xn = XS.ravel()[good]
    yn = YS.ravel()[good]
    meannear = np.nanmean(subim)
    mednear = np.nanmedian(subim)

    return xn, yn, meannear, mednear
    
def reconstruct_model_nparams(trendparams,m,X,Y):
    # given a model and a number of trend parameters, reconstruct the trend
    #
    onemat = np.ones_like(X)
    model = {
    1: lambda m,X,Y,onemat: m[0]*onemat,
    2: lambda m,X,Y,onemat: m[0]*onemat+m[1]*X,
    3: lambda m,X,Y,onemat: m[0]*onemat+m[1]*X+m[2]*Y,
    4: lambda m,X,Y,onemat: m[0]*onemat+m[1]*X+m[2]*Y+m[3]*X**2,
    5: lambda m,X,Y,onemat: m[0]*onemat+m[1]*X+m[2]*Y+m[3]*X**2+m[4]*Y**2,
    6: lambda m,X,Y,onemat: m[0]*onemat+m[1]*X+m[2]*Y+m[3]*X**2+m[4]*Y**2+m[5]*X*Y
    }[trendparams](m,X,Y,onemat) 
    
    return model

def reconstruct_model_nparams2(m, x, y):
    # New reconstruct, just needs m, X, and Y
    # trendparams is inferred from length of m
    # ones are generated based on X
    
    # Figure out terms
    X,Y = np.meshgrid(x,y)
    terms = np.array([np.ones_like(X), X, Y, X**2, Y**2, X*Y])
    
    # Truncate terms and multiply by coefficients, then sum
    model = np.sum(m[:,None] * terms[:len(m)], axis=0)
    
    return model
    
def constrained_lsq(G,d,C,z=0,method='inv'):
    # minimize:   Gm = d 
    # subject to: Cm = z
    # This function works best when the KKT matrix is invertible in the classic sense, and you use the standard 'inv' method
    # Underdetermined/overconstrained problems may fail inelegantly - accurate results with pinv() are not guaranteed!
    #
    # we introduce dummy Lagrange multipliers l, and solve the KKT equations:
    # (from http://stanford.edu/class/ee103/lectures/constrained-least-squares/constrained-least-squares_slides.pdf)
    # [2*G.T*G  C.T] [m] = [2*G.T*d]
    # [C         0 ] [l] = [z]
    #
    #double check the shape of d
    d=np.array(d,ndmin=2).T
    if (np.size(d,0)==1):
        d=d.T
    #double check the shape of C - avoids problem if only one constraint is passed
    C=np.array(C,ndmin=2)
    #form the big matrix -- sorry, these double-parentheses are hard to read
    K=np.column_stack(( np.row_stack(( 2*np.dot(G.T,G),C )),np.row_stack(( C.T,np.zeros(( np.size(C,0),np.size(C,0) )) )) ))
    #form the data column
    if (np.size(z)!=np.size(C,0)):
        z=np.zeros((np.size(C,0),1)) #if z was given incorrectly (or it was left at the default zero)
    D=np.row_stack(( 2*np.dot(G.T,d),z ))
    #compute the model vector
    if(method=='inv'):
        m=np.dot(np.linalg.inv(K),D)
    elif(method=='pinv'):
        m=np.dot(np.linalg.pinv(K),D)
    #separate the model from lagrange multipliers
    mm=m[0:np.size(G,1)]
    return mm
    
def constrained_lsq2(G,d,C,z,method):
    # MY STUFF
    # Useful markers
    nd = np.size(G, 0)
    nt = np.size(G, 1)
    ng = np.size(C, 0)

    # Assemble K matrix
    K = np.zeros((nt+ng, nt+ng))
    K[:nt, :nt] = 2 * np.dot(G.T, G)
    K[:nt, nt:] = C.T
    K[nt:, :nt] = C

    # Assemble D matrix
    D = np.zeros((ng+nt, 1))
    D[:nt] = 2 * np.dot(G.T, d.reshape((nd, 1)))
    D[nt:] = z

    # Check if we're safe to solve
    if np.log10(np.linalg.cond(K)) > 8:
        m, res, rank, sng = np.linalg.lstsq(K, D, None)
    else:
        m = np.linalg.solve(K, D)

    return m[:nt]

def imviz(x,y,z1,z2, sig=2, name='dummy.grd'):
    
    vl = np.nanmean(z1) - sig * np.nanstd(z1)
    vh = np.nanmean(z1) + sig * np.nanstd(z1)
    fig = plt.figure(figsize=(16,8), dpi=192)
    ax = plt.subplot2grid((1,2),(0,0))
    plt.pcolormesh(x, y, z1, shading='auto', norm=Normalize(vl, vh),
                   cmap='Spectral')
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title('Blah')
    
    
    ax = plt.subplot2grid((1,2),(0,1))
    plt.pcolormesh(x, y, z2, shading='auto', norm=Normalize(vl, vh),
                   cmap='Spectral')
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title('Blah')
    
    imname = name.replace('grd', 'png')
    plt.tight_layout()
    # plt.savefig(f'/Users/bruzewskis/Dropbox/bisbasgenmap.png', bbox_inches='tight')
    plt.show()
    
def main():
    # Get data
    # path = '/Users/bruzewskis/Documents/Projects/BISBAS/testing_fulldata/timeseries/'
    path = '/Users/bruzewskis/Downloads/isbas_ground_truth/timeseries_nodetrend/'
    path2 = '/Users/bruzewskis/Downloads/isbas_ground_truth/timeseries_detrended/'
    files = sorted([ f for f in os.listdir(path) if 'ts_mm' in f])
    print(files)
    
    r = 500
    xc = np.random.randint(r, 3900-r)
    yc = np.random.randint(r, 2350-r)
    print(f'{xc-r}:{xc+r}, {yc-r}:{yc+r}')
    
    nanmap = np.zeros((3250,3900), dtype=int)
    nf = 1
    for file in files[nf:nf+1]:
        
        num = None
        # Get my data and detrend
        with netcdf_file(os.path.join(path, file), mode='r') as dat:
            x = dat.variables['lon'][:]
            y = dat.variables['lat'][:]
            im = dat.variables['z'][:]
            
        
        # Get ground truth data
        with netcdf_file(os.path.join(path2, file), mode='r') as dat:
            im_real = dat.variables['z'][:]
            
        
        # Generate some fake GPS points
        gps = np.array([[255.3, 36.6, 10, 0]])
        
        im_corr = detrend_constraints(x, y, im, gps)
        
    #imviz(x,y,im_corr, im_real)
        
    return im_corr, im_real

def get_data_near_h5(file, x0, y0, min_points, max_size=20):
    
    min_size = np.ceil(np.sqrt(min_points)).astype(int)

    with h5py.File(file, 'r') as fo:
        x = fo['x'][:]
        y = fo['y'][:]
        z = fo['displacements']
        X, Y = np.meshgrid(x, y)

        for chunk_size in np.arange(min_size, max_size):

            # This is how we would deal with a non-uniform spacing
            xp = np.interp(x0, x, np.arange(len(x)))
            yp = np.interp(y0, y, np.arange(len(y)))
        
            # Check if the position is outside of image
            if any([xp <= chunk_size, xp >= len(x)-chunk_size,
                    yp <= chunk_size, yp >= len(y-chunk_size)]):
                raise ValueError('This position too close to edge of image')
        
            # Find corners
            xmin = np.ceil( xp - chunk_size/2 ).astype(int)
            ymin = np.ceil( yp - chunk_size/2 ).astype(int)
            xmax = xmin + chunk_size
            ymax = ymin + chunk_size
            
            zarr = z[ymin:ymax, xmin:xmax,:]
            
            cond1 = np.all(np.sum(~np.isnan(zarr), axis=(0,1))>min_points)
            if cond1:
                ym, xm = np.mgrid[ymin:ymax, xmin:xmax]
                xarr = np.broadcast_to(x[xm, None], zarr.shape)
                yarr = np.broadcast_to(y[ym, None], zarr.shape)
                break
    
    return xarr, yarr, zarr

def midhandle(filename, gps, contrained=True, trendparams=3,
              GTG=None, GTd=None):
    
    # Accumulated these
    GTG = 10**np.random.uniform(8,12,(6,6,20))
    GTd = 10**np.random.uniform(8,12,(6,20))
    
    # This is like the file
    x = np.linspace(255.5, 255.6, 310)
    y = np.linspace(32.6, 32.7, 330)
    z = np.random.normal(0, 100, (y.size, x.size, 20))
    z[np.random.choice([True, False], z.shape, p=[0.25, 0.75])] = np.nan
    with h5py.File(filename, 'w') as fo:
        fo['x'] = x
        fo['y'] = y
        fo['displacements'] = z
    
    # Grab the bits
    xg = gps[:,0]
    yg = gps[:,1]
    ng = gps[:,2]
    zg = gps[:,3:]
    
    # Grab data around that point
    G = np.zeros((len(gps), 6, z.shape[-1]))
    d = np.zeros((len(gps), z.shape[-1]))
    for i in range(len(gps)):
        xa, ya, za = get_data_near_h5(filename, xg[i], yg[i], ng[i])
        isgood = ~np.isnan(za)
        numgood = np.sum(isgood, axis=(0, 1))
        Gpoint = np.array([numgood,
                           np.sum(xa, axis=(0, 1), where=isgood),
                           np.sum(ya, axis=(0, 1), where=isgood),
                           np.sum(xa**2, axis=(0, 1), where=isgood),
                           np.sum(ya**2, axis=(0, 1), where=isgood),
                           np.sum(xa*ya, axis=(0, 1), where=isgood)])
        dpoint = (np.nanmean(za, axis=(0, 1)) - zg[i]) * numgood
        
        G[i] = Gpoint
        d[i] = dpoint
    print(G.shape, d.shape)
    return None

if __name__=='__main__':
    # i1, i2 = main()
    np.random.seed(10)
    
    
    # test1
    gps = np.array([[255.52, 32.64, 10, 0]])
    test1 = midhandle('blah.h5', gps)
    
    # test2
    gps = np.column_stack([np.random.uniform(255.51, 255.59, 5),
                           np.random.uniform(32.61, 32.69, 5),
                           np.full(5, 10),
                           np.random.normal(0, 10, (5,1))])
    test2 = midhandle('blah.h5', gps)
    
    # test2
    gps = np.column_stack([np.random.uniform(255.51, 255.59, 5),
                           np.random.uniform(32.61, 32.69, 5),
                           np.full(5, 10),
                           np.random.normal(0, 10, (5,20))])
    test3 = midhandle('blah.h5', gps)
    
    xarr = np.linspace(0, 10, 50)
    yarr = np.linspace(-10, 0, 60)
    tarr = np.arange(555, 559)
    zarr = np.random.random((yarr.size, xarr.size, tarr.size))
    
    with h5py.File('testing.h5', 'w') as fo:
        
        fy = fo.create_dataset('y', data=yarr)
        fy.make_scale('y coordinate')

        fx = fo.create_dataset('x', data=xarr)
        fx.make_scale('x coordinate')

        ft = fo.create_dataset('t', data=tarr)
        ft.make_scale('t coordinate')

        data = fo.create_dataset('displacements', data=np.empty(zarr.shape))

        # Set up scales
        data.dims[0].attach_scale(fy)
        data.dims[0].label = 'lon'
        data.dims[1].attach_scale(fx)
        data.dims[1].label = 'lat'
        data.dims[2].attach_scale(ft)
        data.dims[2].label = 'days'

    dimnames = []
    labels = []
    with h5py.File('testing.h5', 'r') as fz:
        for d in fz['displacements'].dims:
            dimnames.append( d[0].name )
            labels.append( d.label )
        
        print(fz['displacements'].dtype)
    print(dimnames)
    print(labels)
    