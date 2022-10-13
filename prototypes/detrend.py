#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:56:00 2022

@author: bruzewskis
"""

import os
from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.patches import Rectangle


def detrend_constraints(x,y,image,gpsdat,trendparams=3, num=None):
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
    print('dummy1')
    C1, z1 = construct_gpsconstraint(0,image, gpsdat,X, Y)
    print('dummy2')
    C2, z2 = construct_gpsconstraint2(image, gpsdat, x, y)
    print('dummy3')

    # Construct image constraints
    G1, d1 = construct_imconstraint(image, x, y)
    G2, d2 = construct_imconstraint2(image, x, y, num)  

    # select the specified number of trend parameters
    G1 = G1[:, :trendparams]
    G2 = G2[:, :trendparams]
    C1 = C1[:, :trendparams]
    C2 = C2[:, :trendparams]

    # invert for a trend subject to constraint that it matches the value around given GPS points
    m1 = constrained_lsq(G1, d1, C1, z1, method=invmethod)
    m2 = constrained_lsq2(G2, d2, C2, z2, invmethod)

    # reconstruct the model
    model1 = reconstruct_model_nparams(trendparams, m1, X, Y)     
    model2 = reconstruct_model_nparams2(m2, x, y)

    # # Draw it
    # xd = np.linspace(x.min(), x.max(), 10)
    # yd = np.linspace(y.min(), y.max(), 10)
    # XD,YD = np.meshgrid(xd,yd)
    # draw1 = reconstruct_model_nparams(trendparams, m1, XD, YD)
    # draw2 = reconstruct_model_nparams2(m2, xd, yd) 
    # ax = plt.subplot(projection='3d')
    # ax.plot_surface(XD,YD,draw1, color='C2')
    # ax.plot_surface(XD,YD,draw2, color='C3')
    # ax.view_init(30,45+90)
    # plt.show()

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


def construct_imconstraint2(image, x, y, num=None):

    yind, xind = np.where(~np.isnan(image))
    
    if num is not None:
        rc = np.random.randint(0, len(xind), num)
        xind = xind[rc]
        yind = yind[rc]
    
    xG = x[xind]
    yG = y[yind]
    oG = np.ones_like(xG, dtype=np.float)
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
    print('CONSTRUCT CONSTRAINTS')
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
    print('RECONSTRUCT MODEL')
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

def imviz(x,y,z1,z2, sig, name):
    
    vl = np.nanmean(z1) - sig * np.nanstd(z1)
    vh = np.nanmean(z1) + sig * np.nanstd(z1)
    fig = plt.figure(figsize=(10,10), dpi=192)
    ax = plt.subplot2grid((2,2),(0,0))
    plt.pcolormesh(x, y, z1, shading='auto', norm=Normalize(vl, vh),
                   cmap='Spectral')
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title('Mine')
    
    vl = np.nanmean(z2) - sig * np.nanstd(z2)
    vh = np.nanmean(z2) + sig * np.nanstd(z2)
    ax = plt.subplot2grid((2,2),(0,1))
    plt.pcolormesh(x, y, z2, shading='auto', norm=Normalize(vl, vh),
                   cmap='Spectral')
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title('Eric\'s')
    
    diff = z2 - z1
    vs = sig * np.nanstd(diff)
    ax = plt.subplot2grid((2,2),(1,0))
    cb = plt.pcolormesh(x, y, diff, shading='auto', norm=Normalize(-vs, vs),
                   cmap='Spectral')
    
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title('Difference')
    cax = plt.axes([0.54, 0.12, 0.03, 0.35])
    plt.colorbar(cb, cax=cax)
    
    imname = name.replace('grd', 'png')
    # plt.tight_layout()
    plt.savefig(f'outims/{imname}', bbox_inches='tight')
    plt.show()
    
def main():
    # Get data
    path = '/Users/bruzewskis/Documents/Projects/BISBAS/testing_fulldata/timeseries/'
    # path = '/Users/bruzewskis/Downloads/isbas_ground_truth/timeseries_nodetrend/'
    path2 = '/Users/bruzewskis/Downloads/isbas_ground_truth/timeseries_detrended/'
    files = sorted(os.listdir(path))
    
    nanmap = np.zeros((3250,3900), dtype=int)
    for file in files[10:11]:
        print(file)
        
        # Get my data and detrend
        with netcdf.netcdf_file(os.path.join(path, file), mode='r') as dat:
            x = dat.variables['lon'][:]
            y = dat.variables['lat'][:]
            im = dat.variables['z'][:]
        
        # Get ground truth data
        with netcdf.netcdf_file(os.path.join(path2, file), mode='r') as dat:
            im_real = dat.variables['z'][:]
        
        # Generate some fake GPS points
        gps = np.array([[255.3, 36.6, 10, 0]])
        
        imcorr = detrend_constraints(x, y, im, gps)
        plt.hist((imcorr-im_real).ravel(), bins=np.linspace(-30,30), label='All', histtype=u'step')
        plt.legend()
        
        return X,Y,imcorr, im_real
            
if __name__=='__main__':
    Xi,Yi,i1, i2 = main()