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

import bifrost as bf
import bifrost.pipeline as bfp
import cupy as cp

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
        self.np_dtype = bf.dtype.name_nbit2numpy(dcode, nbits)


    def create_reader(self, filename):
        # Log line about reading

        return IntfRead(filename, self.gulp_pixels, self.np_dtype, file_order=self.file_order)

    def on_sequence(self, ireader, filename):

        coord_dtname = np.dtype(self.np_dtype).name
        ireader.xcoords.astype(coord_dtname).tofile('tmp_x.dat')
        ireader.ycoords.astype(coord_dtname).tofile('tmp_y.dat')
        blockslogger.debug(f'{ireader.xcoords.dtype}')

        ohdr = {'name':     filename,
                'gulp':     self.gulp_pixels,
                'zdtype':   coord_dtname,
                'xfile':    'tmp_x.dat',
                'xdtype':   coord_dtname,
                'xname':    ireader.xname,
                'yfile':    'tmp_y.dat',
                'ydtype':   coord_dtname,
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
        self.ref_stack = cp.asarray(ref_stack)

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        ohdr["name"] += "_referenced"

        blockslogger.debug('Started ReferenceBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy()
            odata = ospan.data.as_cupy()

            odata[...] = idata
            odata -= self.ref_stack
            ospan.data[...] = bf.ndarray(odata)# may be unneeded?

        return out_nframe

class GenTimeseriesBlock(bfp.TransformBlock):
    ''' (1,npix,nintf) -> (1,npix,ndates) '''

    def __init__(self, iring, dates, G, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.dates = cp.asarray(dates)
        self.nd = len(dates) #cp
        self.G = cp.asarray(G) #cp

    def on_sequence(self, iseq):

        self.dates.tofile('tmp_t.dat')

        ohdr = deepcopy(iseq.header)
        ohdr['name'] += '_as_ts'
        ohdr['_tensor']['shape'][-1] = self.nd

        ohdr['tfile'] = 'tmp_t.dat'
        ohdr['tdtype'] = self.dates.dtype.name
        ohdr['tname'] = 'time'

        blockslogger.debug('Started GenTimeseriesBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy() 
            odata = ospan.data.as_cupy()

            # Set up matrices to solve
            zdata = idata[0]
            M = ~cp.isnan(zdata)
            A = cp.matmul(self.G.T[None, :, :], M[:, :, None] * self.G[None, :, :]).astype(zdata.dtype)
            B = cp.nansum(self.G.T[:, :, None] * (M*zdata).T[None, :, :], axis=1).T

            # Mask out low-rank values
            lowrank = cp.linalg.matrix_rank(A) != self.nd - 1
            A[lowrank] = cp.eye(self.nd-1)
            B[lowrank] = cp.full(self.nd-1, np.nan)

            # Solve
            model = cp.linalg.solve(A, B)

            # Turn it into a cumulative timeseries
            datediffs = (self.dates - cp.roll(self.dates, 1))[1:]
            changes = datediffs[None,:] * model
            ts = cp.zeros((1,cp.size(zdata,0), self.nd))
            ts[:,:,1:] = cp.cumsum(changes, axis=1)

            odata[...] = ts
            ospan.data[...] = bf.ndarray(odata)

        return out_nframe

class ConvertToMillimeters(bfp.TransformBlock):

    def __init__(self, iring, conv, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.conv = conv

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        ohdr['conversion'] = f'{self.conv}'
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy() 
            odata = ospan.data.as_cupy()

            odata[...] = idata
            odata *= self.conv
            ospan.data[...] = bf.ndarray(odata)# may be unneeded?

        return out_nframe

class WriteAndAccumBlock(bfp.SinkBlock):
    def __init__(self, iring, name, overwrite=True, *args, **kwargs):
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

    def on_sequence(self, iseq):
        # I'm doing a lot of setup here, but this should only be called once

        # Grab header
        hdr = iseq.header

        # Create the axes
        self.yarr = np.fromfile(hdr['yfile'], dtype=hdr['ydtype'])
        fy = self.fo.create_dataset('y', data=self.yarr)
        fy.make_scale('y coordinate')
        os.remove(hdr['yfile'])

        self.xarr = np.fromfile(hdr['xfile'], dtype=hdr['xdtype'])
        fx = self.fo.create_dataset('x', data=self.xarr)
        fx.make_scale('x coordinate')
        os.remove(hdr['xfile'])

        self.tarr = np.fromfile(hdr['tfile'], dtype=hdr['tdtype'])
        ft = self.fo.create_dataset('t', data=self.tarr)
        ft.make_scale('t coordinate')
        os.remove(hdr['tfile'])

        # Generate new data object
        self.shape = ( fy.size, fx.size, ft.size )
        self.imshape = ( fy.size, fx.size )
        blockslogger.debug(f'Here is the shape {self.shape}')
        data = self.fo.create_dataset('displacements', 
                                      data=np.empty(self.shape, 
                                                    dtype=hdr['zdtype']))

        # Set up scales
        data.dims[0].attach_scale(fy)
        data.dims[0].label = hdr['yname']
        data.dims[1].attach_scale(fx)
        data.dims[1].label = hdr['xname']
        data.dims[2].attach_scale(ft)
        data.dims[2].label = hdr['tname']

        # Record gulp, set up buffer
        self.gulp = hdr['gulp']
        self.buffer = np.empty((2*max([self.gulp, fx.size])+1, ft.size), 
                               dtype=hdr['zdtype'])
        self.head = 0
        self.linelen = fx.size
        self.linecount = 0
        blockslogger.debug(f'Generated a buffer of shape {self.buffer.shape}')

        # Set up some stuff for the accumulation (keeping all terms)
        self.GTG = np.zeros((6, 6, ft.size))
        self.GTd = np.zeros((6, ft.size))

    def on_data(self, ispan):

        ### WRITE STUFF ###
        # Put data into the file
        self.buffer[self.head:self.head+self.gulp,:] = ispan.data[0]
        self.head += self.gulp

        # Write out as many times as needed
        while self.head > self.linelen:
            self.fo['displacements'][self.linecount,:,:] = self.buffer[:self.linelen,:]
            self.linecount += 1

            self.head -= self.linelen
            self.buffer = np.roll(self.buffer, -self.linelen, axis=0)

        perc = 100*self.gulp*self.niter/np.product(self.imshape)
        #blockslogger.debug(f'Written {perc:04.1f}% of data')

        ### ACCUMULATE FOR DOTS ###
        # Figure out what the G matrix should look like
        inds = self.niter * self.gulp + np.arange(0, self.gulp)
        yinds, xinds = np.unravel_index(inds, self.imshape)
        xchunk = self.xarr[xinds]
        ychunk = self.yarr[yinds]
        ones = np.ones_like(xchunk)
        G = np.column_stack([ones, xchunk, ychunk, xchunk**2, ychunk**2, xchunk*ychunk])

        # Accumulate dot-product
        """
        I know this looks complicated but I promise it's not so bad. We're
        basically just zero-weighting all the places in the dot product where
        we have bad data in each image. To do this for all images at once
        we take our universal G matrix, do the first half the of dot-product
        (ij,jk->ijk), then we multiply in a boolean weighting and perform the
        summation (ijk,jl->ikl). The G.T*d dot can be done similarly, just with
        a nansum instead of weighting.
        """
        gooddata = ~np.isnan(ispan.data[0])
        self.GTG += np.einsum('ij,jk,jl->ikl', G.T, G, gooddata)
        self.GTd += np.nansum(np.einsum('ij,jk->ijk', G.T, ispan.data[0]), axis=1)
        self.niter += 1

class H5Reader(object):
    '''
    File read object
    '''
    def __init__(self, filename, gulp_size, dtype):

        # Initialize our reader object
        self.step = 0
        self.fo = h5py.File(filename, 'a')
        self.data = self.fo['displacements']
        self.dtype = dtype

        # Double check this gulp-size is acceptable
        self.imshape = (self.fo['y'].size, self.fo['x'].size)
        imsize = np.product(self.imshape)
        self.ndays = self.fo['t'].size
        if imsize%gulp_size==0:
            self.gulp_size = gulp_size
        else:
            raise ValueError('Gulp must evenly divide image size')

        # Generate regions for entire image
        self.regions = np.arange(0, imsize).reshape(-1, self.gulp_size)
        blockslogger.debug(f'Regions have shape {self.regions.shape}')

        # Make a buffer for reading (hdf5 being picky)
        self.linelen = self.fo['x'].size
        bsize = 2*max(self.gulp_size, self.linelen)
        self.buffer = np.zeros((bsize, self.ndays), dtype=self.dtype)
        self.head = 0
        self.linecount = 0

    def read(self):

        try:
            # This will read via the buffer
            while self.head < self.gulp_size:
                self.buffer[self.head:self.head+self.linelen] = self.fo['displacements'][self.linecount]

                self.head += self.linelen
                self.linecount += 1

            out = self.buffer[:self.gulp_size]
            self.head -= self.gulp_size
            self.buffer = np.roll(self.buffer, -self.gulp_size, axis=0)

            return out.astype(self.dtype)
        except IndexError:
            # Catch the index error if we're past the end
            return np.empty((0, self.ndays), dtype=self.dtype)

    def __enter__(self):
        return self

    def close(self):
        pass

    def __exit__(self, type, value, tb):
        self.fo.close()

class ReadH5Block(bfp.SourceBlock):
    """ 
    This guy will read hdf5 files
    """

    def __init__(self, filenames, gulp_pixels, dtype, *args, **kwargs):
        super().__init__(filenames, 1, *args, **kwargs)
        self.dtype = dtype
        self.gulp_pixels = gulp_pixels

        # Do a lookup on bifrost datatype to numpy datatype
        dcode = self.dtype.rstrip('0123456789')
        nbits = int(self.dtype[len(dcode):])
        self.np_dtype = bf.dtype.name_nbit2numpy(dcode, nbits)

    def create_reader(self, filename):
        # Log line about reading

        return H5Reader(filename, self.gulp_pixels, self.np_dtype)

    def on_sequence(self, ireader, filename):
        ndays = int(ireader.ndays)
        ohdr = {'name':     filename,
                'imshape':  str(ireader.imshape),
                '_tensor':  {'dtype':  self.dtype,
                             'shape':  [-1, self.gulp_pixels, ndays],
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

class ApplyModelBlock(bfp.TransformBlock):

    def __init__(self, iring, models, xaxis, yaxis, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.models = cp.asarray(models)
        self.xaxis = cp.asarray(xaxis)
        self.yaxis = cp.asarray(yaxis)

        self.step = 0
        self.ntrend = models.shape[0]
        self.imshape = (yaxis.size, xaxis.size)

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)

        blockslogger.debug('Started ApplyModelBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy() 
            odata = ospan.data.as_cupy()

            # Do some stuff with idata
            gulp_size = cp.size(idata[0],0)
            r_start = self.step * gulp_size
            r_end = (self.step+1) * gulp_size
            yind, xind = cp.unravel_index(cp.arange(r_start, r_end), self.imshape)

            xc = self.xaxis[xind]
            yc = self.yaxis[yind]

            # d(7800) m(3,20) -> c(7800,20)
            ones = cp.full_like(xc, 1)
            raw = cp.column_stack([ones, xc, yc, xc**2, yc**2, xc*yc])
            corr = cp.dot(raw[:,:self.ntrend], self.models)
            #blockslogger.debug(f'On step: {self.step} \n{corr}')
            corr = cp.expand_dims(corr, axis=0)

            self.step += 1

            odata[...] = idata
            odata -= corr
            ospan.data[...] = bf.ndarray(odata)

        return out_nframe

class CalcRatesBlock(bfp.TransformBlock):

    def __init__(self, iring, taxis, deg=1, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.taxis = cp.asarray(taxis)
        self.deg = deg

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        ohdr['_tensor']['shape'] = iseq.header['_tensor']['shape'][:-1]

        blockslogger.debug('Started CalcRatesBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy() 
            odata = ospan.data.as_cupy()

            fits = cp.polyfit(self.taxis, idata[0].T, 1)
            rate = fits[0].reshape((1,-1))

            odata[...] = rate
            ospan.data[...] = bf.ndarray(odata)

        return out_nframe

class AccumRatesBlock(bfp.SinkBlock):

    def __init__(self, iring, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)

        # Set up accumulation
        self.niter = 0

    def on_sequence(self, iseq):

        # Grab useful things from header
        hdr = iseq.header
        self.imshape = eval(hdr['imshape'])
        self.rates = np.zeros(self.imshape)

        blockslogger.debug('Started AccumRatesBlock')

    def on_data(self, ispan):

        s,gulp = ispan.data.shape

        inds = gulp*self.niter + np.arange(0, gulp)
        yinds, xinds = np.unravel_index(inds, self.imshape)

        self.rates[yinds,xinds] = ispan.data[0]
        self.niter += 1

class WriteH5Block(bfp.SinkBlock):

    def __init__(self, iring, filename, dsetname, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.fo = h5py.File(filename, 'a')

        # Set up accumulation
        self.niter = 0

    def on_sequence(self, iseq):

        # Grab useful things from header
        hdr = iseq.header
        span, self.gulp, depth = hdr['_tensor']['shape']

        # Grab useful things from file
        ref_dtype = self.fo['displacements'].dtype
        outshape = (self.fo['displacements'].shape[0], 
                    self.fo['displacements'].shape[1], 
                    depth)
        blockslogger.debug(f'Write block is writing to a {outshape} object')

        # Create dataset
        if 'detrended' in self.fo:
            # should probably verify this is good
            data = self.fo['detrended']
        else:
            data = self.fo.create_dataset('detrended', 
                                          data=np.empty(outshape, 
                                                        dtype=ref_dtype))

        # Assign scales
        for i in range(data.ndim):
            if outshape[i]==1:
                blockslogger.debug('We don\'t need to label this axis')
                continue

            ref_dim = self.fo['displacements'].dims[i]

            data.dims[i].attach_scale(ref_dim[0])
            data.dims[i].label = ref_dim.label

        # Record gulp, set up buffer
        self.linelen = outshape[1]
        self.buffer = np.empty((2*max([self.gulp, self.linelen])+1, depth), 
                               dtype=ref_dtype)
        self.head = 0
        self.linecount = 0

    def on_data(self, ispan):

        ### WRITE STUFF ###
        # Put data into the file
        self.buffer[self.head:self.head+self.gulp,:] = ispan.data[0]
        self.head += self.gulp

        # Write out as many times as needed
        while self.head > self.linelen:
            self.fo['detrended'][self.linecount,:,:] = self.buffer[:self.linelen,:]
            self.linecount += 1

            self.head -= self.linelen
            self.buffer = np.roll(self.buffer, -self.linelen, axis=0)



    