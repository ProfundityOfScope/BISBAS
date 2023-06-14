import h5py
import numpy as np
import os

with h5py.File('ifgramStack.h5', 'r') as fo:

	t0,y0,x0 = fo['coherence'].shape
	attrs = dict(fo.attrs)

	tm,ym,xm = np.array([t0, y0, x0])//2

	for n in range(2,6):
		for m in range(2,6):

			tn = int(t0/2**n)//2
			yn = int(y0/2**m)//2
			xn = int(x0/2**m)//2
			name = f'smallifgs/ifgs_t{2**n:02d}_p{2**m:02d}.h5'

			with h5py.File(name, 'w') as fo2:

				for a in attrs:
					fo2.attrs[a] = attrs[a]

				fo2.attrs['REF_X'] = xm
				fo2.attrs['REF_Y'] = ym

				fo2['coherence'] = fo['coherence'][tm-tn:tm+tn, ym-yn:ym+yn, xm-xn:xm+xn]
				fo2['date'] = fo['date'][tm-tn:tm+tn]
			print(tn, yn, xn, name)