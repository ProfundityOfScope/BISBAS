import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import transforms
import h5py
from time import time
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from scipy import ndimage

import cartopy.io.shapereader as shpreader

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
    print(np.rad2deg(np.arctan2(xp[1,1]-xp[1,0], xp[0,1]-xp[0,0])))
    xo = np.array([[0, float(attrs['WIDTH']), 0],
                   [0, 0, float(attrs['LENGTH'])]])
    Ao = np.row_stack([xo, np.ones(3)])
    Ap = np.row_stack([xp, np.ones(3)])
    M = np.dot(Ap, np.linalg.inv(Ao))
    return M

def make_image(fobj: h5py.File, dname: str, outfile: str = 'image.png', ind: int = 0):
    # Image maker

    # Infer the name of the data
    data = fobj[dname]

    med = np.nanmedian(data)
    scale = np.nanstd(data)*2
    imkwargs = {'vmin': -0.1, 'vmax': 0.1, 'interpolation': 'nearest',
                'origin': 'upper', 'rasterized': True, 'cmap': 'Spectral_r'}

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
    
    tref = trA.transform((fo.attrs['REF_X'], fo.attrs['REF_Y']))
    print(tref)
    ax.scatter(*tref)

    # Plot initial image with limits
    im = ax.imshow(data[ind], transform=tr, **imkwargs)
    fig.colorbar(im, extend='both')
    # ax.set_xlim(np.min(tcorn[:, 0]), np.max(tcorn[:, 0]))
    # ax.set_ylim(np.min(tcorn[:, 1]), np.max(tcorn[:, 1]))
    ax.set_xlim( 106.68, 107)
    ax.set_ylim( -6.38, -6.08)

    if outfile is None:
        return None
    else:
        fig.savefig(outfile, bbox_inches='tight')

tgt = '/Users/bruzewskis/Documents/Projects/BISBAS/tsSlim.h5'
with h5py.File(tgt, 'r') as fo:
    
    make_image(fo, 'rates', outfile=None)