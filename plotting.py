#!/usr/bin/env python3
"""
This file contains various blocks for the Bifrost-ISBAS pipeline
"""

import logging

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import transforms

__author__ = "Seth Bruzewski"
__credits__ = ["Seth Bruzewski", "Jayce Dowell", "Gregory Taylor"]

__license__ = "MIT"
__version__ = "1.0.4"
__maintainer__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__status__ = "development"

plotslogger = logging.getLogger('__main__')
matplotlib.use('Agg')

def stretch_video(fobj: h5py.File, dname: str, outfile: str, fps: int = 10,
                  time: float = 30):
    '''Makes a stretched video.'''


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
    fig, ax = plt.subplots(figsize=(10, 10), dpi=1080/10)
    image = ax.imshow(data[0], cmap='Spectral_r', interpolation='nearest',
                      vmin=med-scale, vmax=med+scale)
    title = ax.set_title(f'{dates[0]:<4.0f} since {date0}')

    # This lets us stretch frames for a uniform video
    n_frames = int(time*fps)
    uniform_dates = np.linspace(np.min(dates), np.max(dates), n_frames)
    frames = np.argmax(uniform_dates[:,None] <= dates[None,:], axis=1)

    # Define update function
    def update(frame):
        # Data
        image.set_data(data[frame])
        title.set_text(f'{uniform_dates[frame]:<4.0f} since {date0}')
        return (image, title)

    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save(outfile, writer='ffmpeg', fps=fps)
    plt.close(fig)

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

def make_image(image, *args, header: dict = None, outfile: str = None,
               vmin: float = None, vmax: float = None, cmap: str = 'Spectral_r',
               interpolation: str = 'nearest', origin: str = 'lower',
               rasterized: bool = True, **kwargs):
    '''Makes an image from a 2D array.'''
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
                   origin=origin, rasterized=rasterized, vmin=vmin, vmax=vmax,
                   *args, **kwargs)
    fig.colorbar(im, extend='both')

    if outfile is None:
        return fig, ax, im
    else:
        fig.savefig(outfile, bbox_inches='tight')


if __name__ == '__main__':
    with h5py.File('timeseries.h5', 'r') as fo:
        make_image(fo['rates'][0], outfile='rates.png', vmin=-0.05, vmax=0.05)
        print('Made rates image')
