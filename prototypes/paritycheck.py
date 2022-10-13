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
from matplotlib.cm import Spectral
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import pickle
from tqdm import trange,tqdm
from time import time

def get_near_data_moore(x, y, image, x0, y0, moore_size=5):

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
    
    if np.isnan(np.nanmedian(subim)):
        raise ValueError
    return np.nanmedian(subim)

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
    return mediannear


def imviz(x, y, z1, z2, vmin, vmax):
    """Use this function to do something."""

    fig = plt.figure(figsize=(10, 5), dpi=175)
    ax = plt.subplot2grid((1, 2), (0, 0), fig=fig)
    plt.pcolormesh(x, y, z1, shading='auto', norm=Normalize(vmin, vmax),
                   cmap=Spectral)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title('Mine')
    
    ax = plt.subplot2grid((1, 2), (0, 1), fig=fig)
    plt.pcolormesh(x, y, z2, shading='auto', norm=Normalize(vmin, vmax),
                   cmap=Spectral)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title('Eric\'s')
    
    # imname = file.replace('grd', 'png')
    plt.tight_layout()
    # plt.savefig(f'outims/{imname}', bbox_inches='tight')
    plt.show()


def invert1(G, data, dates):

    mingood = 0
    nt = len(dates)
    ni, nx, ny = np.shape(data)
    model = np.zeros((nt-1, nx, ny))*np.nan
    Gstore = {}
    for i in range(nx):
        for j in range(ny):
            Igood = np.where(~np.isnan(data[:, i, j]))[0]
            if (len(Igood) > 0):
                Ggood = G[Igood, :]

                cond1 = mingood > 0 and len(Igood) > mingood
                cond2 = mingood <= 0 and np.linalg.matrix_rank(Ggood) == nt-1
                if cond1 or cond2:
                    Ikey = hash(Igood.tobytes())
                    if Ikey not in Gstore:
                        Gstore[Ikey] = np.linalg.pinv(Ggood)
                    Ginv = Gstore[Ikey]
                    model[:, i, j] = np.dot(Ginv, data[Igood, i, j])

    pickpath = '/Users/bruzewskis/Documents/Projects/BISBAS/prototypes/pickled/'
    pickle.dump(model, open(os.path.join(pickpath, 'm4parity.p'), 'wb'))

    return model


def invert2(G, data, dates):

    nt = len(dates) - 1
    ni, nx, ny = data.shape
    datar = data.reshape((ni, nx*ny))

    # Generate matrices
    M = ~np.isnan(datar)
    Tbig = np.matmul(G.T[None, :, :], M.T[:, :, None] * G[None, :, :])
    rbig = np.nansum(G.T[:, :, None] * (M*datar)[None, :, :], axis=1).T

    # Mask out low-rank values
    lowrank = np.linalg.matrix_rank(Tbig) != len(dates) - 1
    Tbig[lowrank] = np.eye(len(dates)-1)
    rbig[lowrank] = np.full(len(dates)-1, np.nan)

    # Solve
    model = np.linalg.solve(Tbig, rbig).T.reshape((nt, nx, ny))

    return model


# Get data
datapath = '/Users/bruzewskis/Documents/Projects/BISBAS/testing_fulldata/intf/'
pickpath = '/Users/bruzewskis/Documents/Projects/BISBAS/prototypes/pickled/'
direcs = sorted(os.listdir(datapath))


G = pickle.load(open(os.path.join(pickpath, 'G4parity.p'), 'rb'))
dates = pickle.load(open(os.path.join(pickpath, 'd4parity.p'), 'rb'))
dfiles = pickle.load(open(os.path.join(pickpath, 'od4parity.p'), 'rb'))


bl = 100
bx = np.random.randint(0,3900-bl)
by = np.random.randint(0,3250-bl)
intfstack = np.zeros((90,bl,bl))
# Get chunks
for i, file in enumerate(dfiles):
    with netcdf.netcdf_file(os.path.join(file), mode='r') as dat:
        x = dat.variables['lon'][bx:bx+bl]
        y = dat.variables['lat'][by:by+bl]
        im = dat.variables['z'][by:by+bl, bx:bx+bl]
        intfstack[i] = im

start = time()
model1 = invert1(G, intfstack, dates)
print(time()-start)

start = time()
model2 = invert2(G, intfstack, dates)
print(time()-start)

X, Y = np.meshgrid(x, y)
ci = 10
imviz(X, Y, model2[ci], model1[ci], -0.25, 0.25)

model1[np.isnan(model1)] = 9999
model2[np.isnan(model2)] = 9999
print(np.allclose(model1, model2, 0.1))

