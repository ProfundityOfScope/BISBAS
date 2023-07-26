#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:51:43 2023

@author: bruzewskis
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.colors import LogNorm
from astropy.time import Time
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from time import sleep, time
import json
import shapely.geometry as sg
import shapely.ops as so
from scipy.interpolate import CubicSpline


def make_image(image, header: dict = None, outfile: str = None, 
               vmin: float = None, vmax: float = None,
               cmap: str = 'Spectral_r', interpolation: str = 'nearest',
               origin: str = 'lower', rasterized: bool = True, 
               *args, **kwargs):
    # Handle extreme bounds
    if vmin is None or vmax is None:
        med = np.nanmedian(image)
        scale = 3*np.nanstd(image)
        vmin = med-scale if vmin is None else vmin
        vmax = med+scale if vmax is None else vmax

    # Set up the figure and axis
    aspect = image.shape[0]/image.shape[1]*0.8
    fig, ax = plt.subplots(figsize=(10, 10*aspect), dpi=1920/10)

    ax.set_xlim(image.shape[1], 0)
    ax.set_ylim(image.shape[0], 0)

    # Plot initial image with limits
    im = ax.imshow(image, cmap=cmap, interpolation=interpolation, 
                   origin=origin, rasterized=rasterized, vmin=vmin, vmax=vmax,
                   *args, **kwargs)
    fig.colorbar(im, extend='both')
    
    return fig, ax, im

def make_video(data, dates):


    r = np.nanstd(data)*5
    
    fps = 30
    time = 30
    nframes = int(fps*time)
    
    
    # Render figure
    fig, ax, im = make_image(data[1], vmin=-r, vmax=r)

    tinterp = np.linspace(np.min(dates), np.max(dates),
                          nframes, endpoint=False)
    
    valid_mask = np.all(~np.isnan(data[1:]), axis=0)
    rows, columns = np.where(valid_mask)
    
    goodpixels = data[:,rows,columns]
    
    cs = CubicSpline(dates, goodpixels)
    emptyimage = np.full_like(data[0], np.nan)
    
    pbar = tqdm(total=nframes)

    # Define update function
    def update(frame):
        # Data
        ti = tinterp[frame]
        im_int = np.copy(emptyimage)
        im_int[rows,columns] = cs(ti)
        im.set_data(im_int)
        pbar.update(1)
        return (im,)

    ani = FuncAnimation(fig, update, frames=nframes, blit=True)
    ani.save('zoomedanim.mp4', writer='ffmpeg', fps=fps)
    plt.close(fig)

if __name__=='__main__':
    
    image = np.load('../cutout.npy')[1:]
    dates = np.load('../datesnum.npy')[1:]
        
    r = 36*2
    make_video(image, dates)
    
    
    # tgt = '/Users/bruzewskis/Downloads/geoBoundaries-IDN-ADM0_simplified.geojson'
    # with open(tgt) as f:
    #     data = json.load(f)
        
    # geometries = [sg.shape(feature['geometry']) for feature in data['features']]

    # coordinates = [(107.75, -7.3), (107.75, -5.6), (105.25, -5.6), (105.25, -7.3)]
    # square = sg.Polygon(coordinates)
    # xs, ys = square.exterior.xy
    # # ax.fill(xs, ys, alpha=0.2)
    
    # water = square.difference(geometries[0] )
    # for geom in water.geoms:
    #     xs, ys = geom.exterior.xy
    #     ax.fill(xs, ys, alpha=0.2, fc='C0', ec='gray')
    #     plt.savefig('/Users/bruzewskis/Dropbox/waterline.png')
    
    # fig.savefig('geographicrates.png', transparent=True)