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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

__author__ = "Seth Bruzewski"
__credits__ = ["Seth Bruzewski", "Jayce Dowell", "Gregory Taylor"]

__license__ = "MIT"
__version__ = "1.0.3"
__maintainer__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__status__ = "development"

plotslogger = logging.getLogger('__main__')

def make_video(data, dates, outfile, fps=10):

    # Set up the figure and axis
    aspect = data.shape[2]/data.shape[1]
    fig, ax = plt.subplots(figsize=(5, 5*aspect), dpi=1080/5)

    imkwargs = {'vmin': -15, 'vmax': 15, 'interpolation': 'nearest',
                'origin': 'lower', 'rasterized': True, 'cmap': 'Spectral_r'}

    def update(frame):
        ax.clear()

        im = ax.imshow(data[frame], **imkwargs)

        return (im,)

    ani = FuncAnimation(fig, update, frames=len(data), blit=True)
    ani.save(outfile, writer='ffmpeg', fps=fps)
    plt.close(fig)


def interp_video(data, dates, outfile, fps=10, nframes=None):

    # Set up the figure and axis
    aspect = data.shape[2]/data.shape[1]
    fig, ax = plt.subplots(figsize=(5, 5*aspect), dpi=1080/5)

    nframes = len(data) if nframes is None else nframes
    tinterp = np.linspace(dates.min(), dates.max(), nframes, endpoint=False)

    imkwargs = {'vmin': -15, 'vmax': 15, 'interpolation': 'nearest',
                'origin': 'lower', 'rasterized': True, 'cmap': 'Spectral_r'}

    def update(frame):
        ax.clear()

        ti = tinterp[frame]
        right = np.argmax(dates > ti)
        left = right - 1

        # Interpolate
        dL = data[left]
        dR = data[right]
        im_int = (dR-dL)/(dates[right]-dates[left]) * (ti-dates[left]) + dL

        im = ax.imshow(im_int, **imkwargs)
        return (im,)

    ani = FuncAnimation(fig, update, frames=nframes, blit=True)
    ani.save(outfile, writer='ffmpeg', fps=fps)
    plt.close(fig)