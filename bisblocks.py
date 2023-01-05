#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains various blocks for the Bifrost-ISBAS pipeline

Created on Sun Aug 14 23:38:03 2022
@author: bruzewskis
"""

import numpy as np
import logging
from copy import deepcopy
from datetime import datetime
import h5py
import os

import bifrost.pipeline as bfp
import bifrost
from bifrost.dtype import name_nbit2numpy

from readers import DataStack

__version__ = 0.1

blockslogger = logging.getLogger('__main__')

class IntfRead(object):
    '''
    File read object
    '''
    def __init__(self, filename, gulp_size, dtype, file_order):

        # Figure out order
        files = [ f'{filename}/{f}' for f in file_order ]
        self.files = files

        # Initialize our reader object
        self.step = 0
        self.reader = DataStack.read(files)
        self.dtype = dtype

        # Keep track of things we'll need to preserve
        self.xcoords = self.reader._xarr
        self.xname = self.reader.xgrd
        self.ycoords = self.reader._yarr
        self.yname = self.reader.ygrd
        self.imshape = self.reader.imshape

        # Double check this gulp-size is acceptable
        imsize = self.reader.imsize
        if imsize%gulp_size==0:
            self.gulp_size = gulp_size
        else:
            raise ValueError('Gulp must evenly divide image size')

        # Generate regions for entire image
        self.regions = np.arange(0, imsize).reshape(-1, self.gulp_size)
        blockslogger.debug(f'Regions have shape {self.regions.shape}')


    def read(self):

        try:
            # We try to read files
            picks = self.regions[self.step]
            d = self.reader[picks]
            self.step += 1

            return d.astype(self.dtype)
        except IndexError:
            # Catch the index error if we're past the end
            return np.empty((0, len(self.files), 3), dtype=self.dtype)

    def __enter__(self):
        return self

    def close(self):
        pass

    def __exit__(self, type, value, tb):
        self.close()
        
class IntfReadBlock(bfp.SourceBlock):
    """ Block for reading binary data from file and streaming it into a bifrost pipeline

    Args:
        filenames (list): A list of filenames to open
        gulp_size (int): Number of elements in a gulp (i.e. sub-array size)
        gulp_nframe (int): Number of frames in a gulp. (Ask Ben / Miles for good explanation)
        dtype (bifrost dtype string): dtype, e.g. f32, cf32
    """
    def __init__(self, filenames, gulp_pixels, dtype, file_order, *args, **kwargs):
        super().__init__(filenames, 1, *args, **kwargs)
        self.dtype = dtype
        self.file_order = file_order
        self.gulp_pixels = gulp_pixels

        # Do a lookup on bifrost datatype to numpy datatype
        dcode = self.dtype.rstrip('0123456789')
        nbits = int(self.dtype[len(dcode):])
        self.np_dtype = name_nbit2numpy(dcode, nbits)


    def create_reader(self, filename):
        # Log line about reading

        return IntfRead(filename, self.gulp_pixels, self.np_dtype, file_order=self.file_order)

    def on_sequence(self, ireader, filename):

        ireader.xcoords.tofile('tmp_x.dat')
        ireader.ycoords.tofile('tmp_y.dat')

        ohdr = {'name':     filename,
                'gulp':     self.gulp_pixels,
                'zdtype':   np.dtype(self.np_dtype).name,
                'xfile':    'tmp_x.dat',
                'xdtype':   ireader.xcoords.dtype.name,
                'xname':    ireader.xname,
                'yfile':    'tmp_y.dat',
                'ydtype':   ireader.ycoords.dtype.name,
                'yname':    ireader.yname,
                '_tensor':  {'dtype':  self.dtype,
                             'shape':  [-1, self.gulp_pixels, len(self.file_order)],
                            },
                }
        return [ohdr]

    def on_data(self, reader, ospans):
        indata = reader.read()

        if indata.shape[0] == self.gulp_pixels:
            ospans[0].data[...] = indata
            return [1]
        else:
            return [0]

class ReferenceBlock(bfp.TransformBlock):
    def __init__(self, iring, ref_stack, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.ref_stack = ref_stack

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        ohdr["name"] += "_referenced"
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        idata = ispan.data
        odata = ospan.data

        odata = idata.copy()
        odata -= self.ref_stack
        return out_nframe

class GenTimeseriesBlock(bfp.TransformBlock):
    ''' (1,npix,nintf) -> (1,npix,ndates) '''

    def __init__(self, iring, dates, G, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.dates = dates
        self.nd = len(dates)
        self.G = G

    def on_sequence(self, iseq):

        self.dates.tofile('tmp_t.dat')

        ohdr = deepcopy(iseq.header)
        ohdr['name'] += '_as_ts'

        ohdr['tfile'] = 'tmp_t.dat'
        ohdr['tdtype'] = self.dates.dtype.name
        ohdr['tname'] = 'time'
        ohdr['_tensor']['shape'][2] = self.nd
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        idata = ispan.data
        odata = ospan.data

        # Set up matrices to solve
        zdata = np.array(idata[0])
        M = ~np.isnan(zdata)
        A = np.matmul(self.G.T[None, :, :], M[:, :, None] * self.G[None, :, :]).astype(zdata.dtype)
        B = np.nansum(self.G.T[:, :, None] * (M*zdata).T[None, :, :], axis=1).T

        print('A', A.shape, np.sum(np.isnan(A))/A.size)
        print(A[1500])
        print('B', B.shape, np.sum(np.isnan(B))/B.size)
        print(B[1500])

        # Mask out low-rank values
        lowrank = np.linalg.matrix_rank(A) != self.nd - 1
        A[lowrank] = np.eye(self.nd-1)
        B[lowrank] = np.full(self.nd-1, np.nan)

        # Solve
        model = np.linalg.solve(A, B)
        print('model', model.shape, np.sum(np.isnan(model))/model.size)
        print(model)

        # Turn it into a cumulative timeseries
        datediffs = (self.dates - np.roll(self.dates, 1))[1:]
        changes = datediffs[None,:] * model
        ts = np.zeros((1,np.size(zdata,0), self.nd))
        ts[:,:,1:] = np.cumsum(changes, axis=1)

        odata[...] = bifrost.ndarray(ts)
        print('timeseries', ts.shape, np.sum(np.isnan(ts))/ts.size)
        print(ts)
        return out_nframe

class WriteAndAccumBlock(bfp.SinkBlock):
    def __init__(self, iring, name, overwrite=True, trendparams=3, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)

        if os.path.exists(name):
            if overwrite:
                blockslogger.debug('Overwriting existing file')
                os.remove(name)
            else:
                blockslogger.error('File already exists, try overwrite=True')
                raise OSError('File already exists, try overwrite=True')

        # Open file
        self.fo = h5py.File(name, mode='x')

        # Set up accumulation
        self.niter = 0
        self.trendparams = trendparams

    def on_sequence(self, iseq):
        # I'm doing a lot of setup here, but this should only be called once

        # Grab header
        hdr = iseq.header

        # Create the axes
        self.tarr = np.fromfile(hdr['tfile'], dtype=hdr['tdtype'])
        ft = self.fo.create_dataset('t', data=self.tarr)
        ft.make_scale('t coordinate')
        os.remove(hdr['tfile'])

        self.xarr = np.fromfile(hdr['xfile'], dtype=hdr['xdtype'])
        fx = self.fo.create_dataset('x', data=self.xarr)
        fx.make_scale('x coordinate')
        os.remove(hdr['xfile'])

        self.yarr = np.fromfile(hdr['yfile'], dtype=hdr['ydtype'])
        fy = self.fo.create_dataset('y', data=self.yarr)
        fy.make_scale('y coordinate')
        os.remove(hdr['yfile'])

        # Generate new data object
        self.shape = ( ft.size, fy.size, fx.size )
        self.imshape = ( fy.size, fx.size )
        blockslogger.debug(f'Here is the shape {self.shape}')
        data = self.fo.create_dataset('displacements', data=np.empty(self.shape, dtype=hdr['zdtype']))

        # Set up scales
        data.dims[0].attach_scale(ft)
        data.dims[0].label = hdr['tname']
        data.dims[1].attach_scale(fy)
        data.dims[1].label = hdr['yname']
        data.dims[2].attach_scale(fx)
        data.dims[2].label = hdr['xname']

        # Record gulp, set up buffer
        self.gulp = hdr['gulp']
        self.buffer = np.empty((ft.size, 2*max([self.gulp, fx.size])+1), dtype=hdr['zdtype'])
        self.head = 0
        self.linelen = fx.size
        self.linecount = 0
        blockslogger.debug(f'Generated a buffer of shape {self.buffer.shape}')

        # Set up some stuff for the accumulation
        self.GTG = np.zeros((self.trendparams, self.trendparams, ft.size))
        self.GTd = np.zeros((self.trendparams, ft.size))

    def on_data(self, ispan):

        ### WRITE STUFF ###
        # Put data into the file
        blockslogger.debug(f'Writing {self.gulp} values to disk, head at {self.head}')
        self.buffer[:,self.head:self.head+self.gulp] = ispan.data[0].T
        self.head += self.gulp

        # Write out as many times as needed
        while self.head > self.linelen:
            self.fo['displacements'][:,self.linecount] = self.buffer[:,:self.linelen]
            self.linecount += 1

            self.head -= self.linelen
            self.buffer = np.roll(self.buffer, -self.linelen, axis=1)

        perc = 100*self.gulp*self.niter/np.product(self.imshape)
        blockslogger.debug(f'Written {perc:04.1f}% of data')

        ### ACCUMULATE FOR DOTS ###
        # Figure out what the G matrix should look like
        inds = self.niter * self.gulp + np.arange(0, self.gulp)
        yinds, xinds = np.unravel_index(inds, self.imshape)
        xchunk = self.xarr[xinds]
        ychunk = self.yarr[yinds]
        ones = np.ones_like(xchunk)
        Gfull = np.column_stack([ones, xchunk, ychunk, xchunk**2, ychunk**2, xchunk*ychunk])
        G = Gfull[:,:self.trendparams]

        # Do the dot products and whatnot
        gooddata = ~np.isnan(ispan.data[0])
        self.GTG += np.einsum('ij,jk,jl->ikl', G.T, G, gooddata)
        self.GTd += np.nansum(np.einsum('ij,jk->ijk', G.T, ispan.data[0]), axis=1)
        self.niter += 1



    