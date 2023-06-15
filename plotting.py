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


def make_video(fobj: h5py.File, outfile: str, fps: int = 10,
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
    # Infer the name of the data
    for name in ['detrended', 'timeseries', 'coherence']:
        if name in fobj:
            break
    data = fo[name]

    # Extract some data info
    dates = fo['datenum']
    date0 = fo['datestr'][0].decode()
    date0 = f'{date0[:4]}-{date0[4:6]}-{date0[6:]}'

    scale = np.nanstd(data)
    imkwargs = {'vmin': -scale, 'vmax': scale, 'interpolation': 'nearest',
                'origin': 'lower', 'rasterized': True, 'cmap': 'Spectral_r'}

    # Set up the figure and axis
    aspect = data.shape[1]/data.shape[2]*0.8
    fig, ax = plt.subplots(figsize=(10, 10*aspect), dpi=1080/10)

    # Get the Affine transform
    M = findAffineMeta(fobj.attrs)
    trA = transforms.Affine2D(M)
    tr = trA + ax.transData

    # Find corners
    corners = np.array([[0, data.shape[2], 0, data.shape[2]],
                        [0, 0, data.shape[1], data.shape[1]]]).T
    tcorn = trA.transform(corners)

    # Plot initial image with limits
    im = ax.imshow(data[0], transform=tr, **imkwargs)
    fig.colorbar(im, extend='both')
    ax.set_xlim(np.min(tcorn[:, 0]), np.max(tcorn[:, 0]))
    ax.set_ylim(np.min(tcorn[:, 1]), np.max(tcorn[:, 1]))

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


if __name__=='__main__':

    tgt = 'ts.h5'
    with h5py.File(tgt, 'r') as fo:
        # make_video(fo, 'test.mp4', 5)
        make_video(fo, 'testint.mp4', 20, 100)
