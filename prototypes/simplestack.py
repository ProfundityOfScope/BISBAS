import numpy as np
import matplotlib.pyplot as plt

import bifrost.pipeline as bfp
from bifrost.dtype import name_nbit2numpy
from datetime import datetime

def StackRead():
	def __init__(self, dirname, gulp_size, dtype, forder):

		self.file_objs = []
		for file in forder:
			fo = np.memmap(file, dtype=dtype, mode='r')
			self.file_objs.append(fo)

		self.dtype = dtype
		self.gulp_size = gulp_size
		self.step = 0

		self.regions = np.arange(0, fo.size).reshape(-1, gulp_size)

	def read(self):
		
		outdata = np.empty((self.gulp_size, len(self.file_objs)))

		for i,fo in enumerate(self.file_objs):
			outdata[:,i] = fo[self.regions[self.step]]

		self.step += 1
		return outdata

	def __enter__(self):
		return self

	def close(self):
		pass

	def __exit__(self, type, value, tb):
		self.close()

class StackReadBlock(bfp.SourceBlock):
	""" Block for reading binary data from file and streaming it into a bifrost pipeline

	Args:
		filenames (list): A list of filenames to open
		gulp_size (int): Number of elements in a gulp (i.e. sub-array size)
		gulp_nframe (int): Number of frames in a gulp. (Ask Ben / Miles for good explanation)
		dtype (bifrost dtype string): dtype, e.g. f32, cf32
	"""
	def __init__(self, filenames, gulp_pixels, dtype, file_order, *args, **kwargs):
		super().__init__(filenames, gulp_pixels, *args, **kwargs)
		self.dtype = dtype
		self.file_order = file_order
		self.gulp_pixels = gulp_pixels

	def create_reader(self, filename):
		# Log line about reading
		# Do a lookup on bifrost datatype to numpy datatype
		dcode = self.dtype.rstrip('0123456789')
		nbits = int(self.dtype[len(dcode):])
		np_dtype = name_nbit2numpy(dcode, nbits)

		return StackRead(filename, self.gulp_pixels, np_dtype, forder=self.file_order)

	def on_sequence(self, ireader, filename):
		ohdr = {'name': filename,
				'_tensor': {
						'dtype':  self.dtype,
						'shape':  [-1, len(self.file_order)],
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

class PrintStuffBlock(bfp.SinkBlock):
	def __init__(self, iring, *args, **kwargs):
		super(PrintStuffBlock, self).__init__(iring, *args, **kwargs)
		self.n_iter = 0

	def on_sequence(self, iseq):
		print("[%s]" % datetime.now())
		print(iseq.name)
		pprint(iseq.header)
		self.n_iter = 0

	def on_data(self, ispan):
		now = datetime.now()
		if self.n_iter % 100 == 0:
			print("[%s] %s" % (now, ispan.data))
		self.n_iter += 1

if __name__=='__main__':

	import os

	path = os.path.join(os.getcwd(), 'fakeims')
	files = sorted([ os.path.join(path, f) for f in os.listdir(path) ])

	with bfp.get_default_pipeline() as PIPELINE1:

		b_read = StackReadBlock([path], 1000, 'f64', files)
		b_out = PrintStuffBlock(b_read)

		PIPELINE1.run()



