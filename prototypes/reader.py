import numpy as np
import os
from scipy.io import netcdf

class DataStack(object):
    '''
    Lets see if we can make an object which can abstract away the headache of
    file access
    '''
    
    def __init__(self, directory):
        self._shape = (3,4950,10_000,10_000)
    
    def __getitem__(self, key):
        print(val)
        
    def __setitem__(self, key, value):
        print(val)
        
    @property
    def shape(self):
        return self._shape

class intfRead(object):
	'''
	A prototype reader object, looks inside a directory (filename) and grabs all the
	files inside of it, then on each read dumps some chunk of all those files into an
	array object which is shaped like (3, numfiles, numpixels), where the first dim 
	is set up like 'xyz'. 
	'''
	def __init__(self, filename, gulp_size, dtype):
		self.step = 0

		self.file_objs = []
		# This would actually need to be in some sort of order
		for r,d,f in os.walk(filename):
			for file in f:
				if file.endswith('grd'):
					self.file_objs.append( netcdf.netcdf_file(os.path.join(r,file)) )

		# Check if gulps nicely divide image
		imsize = self.file_objs[0].variables['x'].shape[0]**2
		if imsize%gulp_size==0:
			self.gulp_size = gulp_size
		else:
			raise ValueError('Gulp must evenly divide image size')

		self.dtype = dtype

	def read(self):
		picks = np.arange(self.step*self.gulp_size, (self.step+1)*self.gulp_size)
		d = np.zeros((3, len(self.file_objs), self.gulp_size), dtype=self.dtype)

		for i in range(len(self.file_objs)):
			file = self.file_objs[i]
			x = file.variables['x']
			y = file.variables['y']
			z = file.variables['z']

			xt = x[picks%x.shape[0]].copy()
			yt = y[picks//x.shape[0]].copy()

			zind = np.unravel_index(picks, z.shape)
			zt = z[zind].copy()

			d[0,i] = xt
			d[1,i] = yt
			d[2,i] = zt

		self.step += 1
		return d

if __name__=='__main__':
	a = intfRead('/Users/bruzewskis/Documents/Projects/bifrost_isbas/isbas/test/intf/', 100, np.float32)
	test1 = a.read()
	test2 = a.read()
	print(test1.shape, test2.shape, np.allclose(test1[2], test2[2]))