#!/usr/bin/env python3
"""
This file contains various blocks for the Bifrost-ISBAS pipeline
"""

import os
import sys
import time
import logging
from datetime import datetime

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import transforms

__author__ = "Seth Bruzewski"
__credits__ = ["Seth Bruzewski", "Jayce Dowell", "Gregory Taylor"]

__license__ = "MIT"
__version__ = "1.0.3"
__maintainer__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__status__ = "development"

plotslogger = logging.getLogger('__main__')
matplotlib.use('Agg')


def findAffineMeta(attrs: dict):
    """
    Make an Affine matrix.

    Use information in the hdf5 attributes to generate an appropriate Affine
    transformation. We're transforming 3 corners of the image.

    Parameters
    ----------
    attrs : dict
        Attributes of HDF5 file.

    Returns
    -------
    M : np.array
        Affine transformation matrix.

    """
    xp = np.array([[float(attrs[f'LON_REF{i+1}']),
                    float(attrs[f'LAT_REF{i+1}'])] for i in range(3)]).T
    xo = np.array([[0, float(attrs['WIDTH']), 0],
                   [0, 0, float(attrs['LENGTH'])]])
    Ao = np.row_stack([xo, np.ones(3)])
    Ap = np.row_stack([xp, np.ones(3)])
    M = np.dot(Ap, np.linalg.inv(Ao))
    return M

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
    fig, ax = plt.subplots(figsize=(10, 10*aspect), dpi=1080/10)

    # Get the Affine transform
    if header is not None:
        M = findAffineMeta(header)
        trA = transforms.Affine2D(M)
        tr = trA + ax.transData

        # Find corners
        corners = np.array([[0, image.shape[1], 0, image.shape[1]],
                            [0, 0, image.shape[0], image.shape[0]]]).T
        tcorn = trA.transform(corners)
        ax.set_xlim(np.min(tcorn[:, 0]), np.max(tcorn[:, 0]))
        ax.set_ylim(np.min(tcorn[:, 1]), np.max(tcorn[:, 1]))
    else:
        tr = ax.transData
        ax.set_xlim(image.shape[1], 0)
        ax.set_ylim(image.shape[0], 0)

    # Plot initial image with limits
    im = ax.imshow(image, transform=tr, cmap=cmap, interpolation=interpolation, 
                   origin=origin, rasterized=rasterized, *args, **kwargs)
    fig.colorbar(im, extend='both')

    if outfile is None:
        return fig, ax, im
    else:
        fig.savefig(outfile, bbox_inches='tight')


def make_video(fobj: h5py.File, dname: str, outfile: str, fps: int = 10,
               nframes: int = None):
    """
    Make a video.

    Uses the output of the BISBAS pipeline to generate a pretty video that
    does all the hard work for you.

    Parameters
    ----------
    fobj : h5py.File
        The output hdf5 file, ideally containing the detrended timeseries.
    outfile : str
        The name of the file to write to. Should include the extension.
    fps : int, optional
        Frames per second. The default is 10.
    nframes : int, optional
        Number of frames to interpolate to. The default is None, which will
        produce no interpolation.

    Returns
    -------
    im : TYPE
        DESCRIPTION.

    """
    # Shortcut name the data
    data = fobj[dname]
    header = dict(fobj.attrs)

    # Extract some data info
    dates = fobj['datenum']
    date0 = fobj['datestr'][0].decode()
    date0 = f'{date0[:4]}-{date0[4:6]}-{date0[6:]}'

    # Calculate some things
    med = np.nanmedian(data)
    scale = 3*np.nanstd(data)

    # Render figure
    fig, ax, im = make_image(data[0], header, vmin=med-scale, vmax=med+scale)

    # Need these if we interpolate
    interpolate = nframes is not None
    nframes = len(data) if not interpolate else nframes
    tinterp = np.linspace(np.min(dates), np.max(dates),
                          nframes, endpoint=False)

    # Define update function
    def update(frame):
        # Data
        if not interpolate:
            im.set_data(data[frame])
        else:
            ti = tinterp[frame]
            ri = np.argmax(dates > ti)
            td = (ti - dates[ri])/(dates[ri]-dates[ri-1])
            im_int = (data[ri]-data[ri-1]) * td + data[ri]
            im.set_data(im_int)

        # Title
        date = tinterp[frame] if interpolate else dates[frame]
        ax.set_title(f'{date:<4.0f} since {date0}')
        return (im,)

    ani = FuncAnimation(fig, update, frames=nframes, blit=True)
    ani.save(outfile, writer='ffmpeg', fps=fps)
    plt.close(fig)


if __name__ == '__main__':
    outfile = 'timeseries.h5'
    with h5py.File(outfile, 'r') as fo:
        make_image(fo['rates'][0], outfile='rates.png')
        print('Made rates image')
        make_video(fo, 'timeseries', 'raw_ts.mp4', 5) #5
        print('Made raw data animation')
        make_video(fo, 'detrended', 'raw_ramp.mp4', 5) #5
        print('Made raw data animation')
