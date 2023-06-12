import h5py
import numpy as np
import os

with h5py.File('ifgramStack.h5', 'r') as fp:

	t0,y0,x0 = fo['coherence'].shape
	attrs = dict(fo.attrs)

	for n in range(4):

		tn = int(t0/2**n)
		yn = int(y0/2**n)
		xn = int(x0/2**n)

		print(tn, yn, xn)