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

def make_video(data, dates, outfile, fps=10, interpolate=False, nframes=100):

	# Set up the figure and axis
	fig, ax = plt.subplots(figsize=(5,5), dpi=1080/5)

	# Might need this
	if interpolate:
		tinterp = np.linspace(t.min(), t.max(), nframes, endpoint=False)
	else:
		nframes = len(data)

	imkwargs = {'vmin': 0, 'vmax': 6000, 'interpolation': 'nearest', 
				'origin': 'lower', 'rasterized': True}

	def update(frame):
	    ax.clear()
	    
	    if interpolate:
		    ti = tinterp[frame]
		    right = np.argmax(t>ti)
		    left = right - 1
		    im_int = (data[right]-data[left])/(dates[right]-dates[left]) * (ti-dates[left]) + data[left]
		    im = ax.imshow(im_int, **imkwargs)
		else:
			im = ax.imshow(data[frame], **imkwargs)
	    return (im,)
	    
	ani = animation.FuncAnimation(fig, update, frames=nframes, blit=True)
	ani.save(outfile, writer='ffmpeg', fps=fps)
	plt.close(fig)