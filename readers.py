#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temp text up here

Created on Thu Jul 14 12:22:08 2022
@author: bruzewskis
"""
from __future__ import annotations

import os
import sys
import logging
import numpy as np
from scipy.io import netcdf

__all__ = ['DataStack', 'read_baselines', 'read_igram_ids']
__version__ = 0.1
__author__ = 'Seth Bruzewski'

readerslogger = logging.getLogger('__main__')

class DataStack():
    '''
    This class abstracts away file handling. Given a directory of interferograms
    it will read them all the a memory map, then selectively extract and format
    requested data for you
    '''

    def __init__(self, file_objs: list, mode='r') -> None:
        '''
        Generate the class from available files provided.

        Parameters
        ----------
        file_objs : list
            An ordered list of file objects we'll be reading from.

        Raises
        ------
        KeyError
            If we cannot infer coordinate convention from file.

        '''
        self.file_objs = file_objs

        # Figure out image properties
        self.imshape = self.file_objs[0].variables['z'].shape #unsafe if empty
        self.imsize = np.product(self.imshape)
        readerslogger.debug(f'Found {len(self.file_objs)} files with shape {self.imshape}')

        # Establish grid convention
        dims = self.file_objs[0].dimensions
        if 'lon' in dims and 'lat' in dims:
            readerslogger.debug('Lon/Lat coordinates inferred from data')
            self.xgrd = 'lon'
            self.ygrd = 'lat'
        elif 'x' in dims and 'y' in dims:
            readerslogger.debug('X/Y coordinates inferred from data')
            self.xgrd = 'x'
            self.ygrd = 'y'
        else:
            readerslogger.debug('File gridding does not match any known standard')
            raise KeyError('File gridding does not follow any known standard')

        # Figure out object properties
        self.shape = (len(self.file_objs), self.imsize, 3)
        self.size = np.product(self.shape)
        readerslogger.debug(f'Images will be treated as a {self.shape} object')

        # Save some data for later use
        self._xarr = self.file_objs[0].variables[self.xgrd][:].copy()
        self._yarr = self.file_objs[0].variables[self.ygrd][:].copy()

    @classmethod
    def read(cls, files: list) -> DataStack:
        '''
        This classmethod should be used to read in an ordered list of grid
        files. The order will be preserved in the order of the final array
        the user can access

        Parameters
        ----------
        files : list
            An ordered list of files.

        Returns
        -------
        new_stack: DataStack
            A new DataStack object created from the files.

        '''

        file_objs = []
        for file in files:
            file_obj = netcdf.netcdf_file(file, 'r') #mmap
            file_objs.append(file_obj)

        new_stack = cls(file_objs)
        return new_stack

    @classmethod
    def empty_like(cls, other: DataStack, directory: str, dates: np.ndarray) -> DataStack:
        '''TODO: Make this work, not sure why it doesn't, some weird key error'''

        # Everything we need for X
        kx = other.xgrd
        nx = other._xarr.size
        lx = kx if kx=='x' else 'longitude'
        ux = None if kx=='x' else 'degrees_east'
        x_f32 = other._xarr.astype(np.float32)
        x_ar = np.float32([other._xarr.min(), other._xarr.max()])

        # Everything we need for Y
        ky = other.ygrd
        ny = other._yarr.size
        ly = ky if ky=='y' else 'latitude'
        uy = None if ky=='y' else 'degrees_north'
        y_f32 = other._yarr.astype(np.float32)
        y_ar = np.float32([other._yarr.min(), other._yarr.max()])
        readerslogger.debug(f'{nx} and {ny} and {kx} and {ky}')

        # Everything we need for Z
        z_f32 = np.full((ny,nx), np.nan, dtype=np.float32)
        z_ar = np.float32([-999,999])

        # Loop over files we need to create
        os.makedirs(directory, exist_ok=True)
        file_objs = []
        for i, date in enumerate(dates):
            fname = f'{directory}/ts_mm_{date:04d}.grd'
            file = netcdf.netcdf_file(fname, 'w')

            # Set up meta-data and whatnot
            file.title = 'Created using scipy.io.netcdf'

            # Set of axes
            file.node_offset = 0
            file.Conventions = 'COARDS/CF-1.0'
            file.createDimension(kx, nx)
            file.createDimension(ky, ny)
            xvar = file.createVariable(kx, np.float32, (kx,))
            yvar = file.createVariable(ky, np.float32, (ky,))
            zvar = file.createVariable('z', np.float32, (ky, kx))

            # Meta variable info
            xvar.long_name = lx
            yvar.long_name = ly
            xvar.units = ux
            yvar.units = uy

            # Assignments
            xvar[:] = x_f32
            yvar[:] = y_f32
            zvar[:] = z_f32
            
            xvar.actual_range = x_ar
            yvar.actual_range = y_ar
            zvar.actual_range = z_ar

            readerslogger.debug(f'Generated empty file ts_mm_{date:04d}.grd')
            file_objs.append(file)

        new_stack = cls(file_objs)
        return new_stack

    def __getitem__(self, key: np.ndarray | slice ) -> np.ndarray:
        '''
        This is a specialized getitem, with the sole purpose of extracting
        strips from the data. It will yell at you if you try to extract
        anything other than strips. Numpy arrays are preferred as keys, so
        slices will be converted

        Parameters
        ----------
        key : np.ndarray | slice
            The desired key to be extracted.

        Raises
        ------
        IndexError
            If key is not one of the things we can handle.

        Returns
        -------
        data : np.ndarray
            Data corresponding to requested keys.

        '''

        # Evaluate key to be safe
        if isinstance(key, np.ndarray):
            # All good, we like arrays
            newkey = key
        elif isinstance(key, slice):
            # Slices are a pain with how we index, convert
            newkey = np.arange(self.imsize)[key]
        else:
            # Yell if we have to
            readerslogger.error('__getitem__ only accepts slices or arrays')
            raise IndexError('__getitem__ only accepts slices or arrays')

        data = np.empty((np.size(self, 0), len(newkey), np.size(self,2)))
        for i, file in enumerate(self.file_objs):
            x = file.variables[self.xgrd]
            y = file.variables[self.ygrd]
            z = file.variables['z']

            xt = x[newkey%x.shape[0]].copy()
            yt = y[newkey//x.shape[0]].copy()

            zind = np.unravel_index(newkey, z.shape)
            zt = z[zind].copy()

            data[i,:,0] = xt
            data[i,:,1] = yt
            data[i,:,2] = zt

        return data

    def __setitem__(self, key: np.ndarray, value: float) -> None:
        '''TODO: Use this to dump data out to files '''

        # Evaluate key to be safe
        if isinstance(key, np.ndarray):
            # All good, we like arrays
            newkey = key
        elif isinstance(key, slice):
            # Slices are a pain with how we index, convert
            newkey = np.arange(self.imsize)[key]
        else:
            # Yell if we have to
            readerslogger.error('__getitem__ only accepts slices or arrays')
            raise IndexError('__getitem__ only accepts slices or arrays')

        for i, file in enumerate(self.file_objs):
            
            z = self.file_objs[i].variables['z']
            zind = np.unravel_index(newkey, z.shape)
            z[zind] = value[i]

    def data_near(self, x0: float, y0: float, chunk_size: int) -> np.ndarray:
        '''
        Given a coordinate of interest, this function will extract a square
        chunk of data around that point, mostly leveraging other methods for
        efficiency. Note that `chunk_size` here is the side length of the
        square, so a chunk size of 8 will return 64 pixels.

        Parameters
        ----------
        x0 : float
            The x/lon coordinate.
        y0 : float
            The y/lat coordinate.
        chunk_size : int
            The edge length of the square chunk to extract.

        Returns
        -------
        subdata : np.ndarray
            The requested data around (x0,y0).

        '''

        # Extract data using coords method
        coords_rav = self.coords_near(x0, y0, chunk_size)
        subdata = self.__getitem__(coords_rav)

        return subdata

    def coords_near(self, x0: float, y0: float, chunk_size: int) -> np.ndarray:
        '''
        Given a coordinate of interest, this function will find the coordinates
        of the nearest small square chunk of pixels. These coordinates are then
        raveled such that they can access the raveled version of the data a
        DataStack object behaves as.

        Parameters
        ----------
        x0 : float
            The x/lon coordinate.
        y0 : float
            The y/lat coordinate.
        chunk_size : int
            The edge length of the square chunk to extract.

        Raises
        ------
        ValueError
            If requested point is outside of the data.

        Returns
        -------
        coords_rav : np.ndarray
            Array of coordinates for the square, raveled to index DataStack.

        '''

        # This is how we would deal with a non-uniform spacing
        xp = np.interp(x0, self._xarr, np.arange(len(self._xarr)))
        yp = np.interp(y0, self._yarr, np.arange(len(self._yarr)))
        coord_str = f'x:{xp:.0f}, y:{yp:.0f}'
        readerslogger.debug(f'Requested reference around pixel {coord_str}')

        # Check if the position is outside of image
        if any([xp <= chunk_size, xp >= len(self._xarr)-chunk_size,
                yp <= chunk_size, yp >= len(self._yarr)-chunk_size ]):
            readerslogger.error('This position too close to edge of image')
            raise ValueError('This position too close to edge of image')

        # Find corners
        xmin = np.ceil( xp - chunk_size/2 ).astype(int)
        ymin = np.ceil( yp - chunk_size/2 ).astype(int)
        xmax = xmin + chunk_size
        ymax = ymin + chunk_size

        # Extracts data inside those corners
        xind = np.arange(xmin, xmax)
        yind = np.arange(ymin, ymax)
        coords = np.array(np.meshgrid(xind, yind)).reshape(2,-1)
        coords_rav = np.ravel_multi_index( coords, self.imshape )

        return coords_rav

    def find_best_chunk_size(self, x0: float, y0: float, num: int) -> int:
        '''
        Search around a reference point such that we find a square arround it
        with more than `num` non-NaN pixels in every file. Useful for
        setting up other methods DataStack has.

        Parameters
        ----------
        x0 : float
            The x/lon coordinate.
        y0 : float
            The y/lat coordinate.
        num : int
            Number of non-NaN pixels desired.

        Raises
        ------
        ValueError
            Raised if best chunk size would be greater than `max_size`.

        Returns
        -------
        best:  int
            The identified best chunk size we've found.

        '''

        max_size = 20

        min_size = np.ceil(np.sqrt(num)).astype(int)
        best = min_size
        for chunk_size in np.arange(min_size, max_size):
            subdata = self.data_near(x0, y0, chunk_size)

            # Check to see if we have enough good pixels
            num_good_per_date = np.sum(~np.isnan(subdata[:,:,2]), axis=1)
            mngpd = np.min(num_good_per_date)
            readerslogger.debug(f'chunk_size={chunk_size} worst intf has {mngpd} valid pixels')
            if np.all(num_good_per_date >= num):
                best = chunk_size
                break
        else:
            raise ValueError('Not enough good pixels near this point')
            
        return best


    def __del__(self):
        for file in self.file_objs:
            file.close()

def read_baselines(fname):
    '''
    This code taken from original code
    read the baseline table and store in date-sorted order
    '''

    readerslogger.info('Reading some shit')
    strdat=np.genfromtxt(fname,str,usecols=0)
    numdat=np.genfromtxt(fname,usecols=(1,2,4))
    sortorder=np.argsort(numdat[:,1])

    ids=strdat[sortorder]
    jdates=np.array([str(int(np.floor(i))) for i in numdat[sortorder,0]])
    dates=numdat[sortorder,1].astype(int)
    bperp=numdat[sortorder,2]

    return ids,jdates,dates,bperp

def read_igram_ids(sat, fname, ids):
    '''
    This code taken from original code, modified
    read intf.in and convert to a list of tuples of IDs
    '''

    sat_map = {'ALOS': slice(13,18), 'ALOS2': slice(12,17), 'S1': slice(0,18)}
    if not sat in sat_map:
        readerslogger.error(f'Error: satellite {sat} not yet implemented.')
        sys.exit(1)

    igrams=np.genfromtxt(fname,dtype=str)
    igram_ids=[]
    sat_slice = sat_map[sat]
    igram_ids = np.zeros((len(igrams),2), dtype=int)
    for i, igram in enumerate(igrams):
        igramsplit=igram.split(':')
        strid0=igramsplit[0][sat_slice]
        strid1=igramsplit[1][sat_slice]
        id0=np.where(ids==strid0)[0][0]
        id1=np.where(ids==strid1)[0][0]
        igram_ids[i] = np.array([id0, id1])
    return igram_ids

def read_wavelength_conversion(prmfile: str) -> float:
    with open(prmfile) as f:
        for line in f:
            if 'wavelen' in line:
                wavelen=float(line.split()[2])
    conv = (-1000)*wavelen/(4*np.pi)
    readerslogger.debug(f'Wavelength is {wavelen}')
    return conv

def stack_read_test():
    ''' Dummy doc string '''
    tgt = '/Users/bruzewskis/Documents/Projects/BISBAS/testing/intf/'
    files = []
    for r,_,f in os.walk(tgt):
        for file in f:
            if file.endswith('.grd'):
                path = os.path.join(r, file)
                files.append(path)

    b = DataStack.read(files)
    print(f'Read in a stack with shape {b.shape}')

    N = 10
    xr = np.random.normal(255.3, 0.1, N)
    yr = np.random.normal(36.6, 0.1, N)

    rads = np.zeros(N, dtype=int)
    for i in range(N):
        rads[i] = b.find_best_chunk_size(xr[i], yr[i], 10)
    print(f'Minimum radius we\'d need is {max(rads)}')
    

if __name__=='__main__':
    stack_read_test()
