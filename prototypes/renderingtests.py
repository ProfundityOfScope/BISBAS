import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import transforms
import h5py
from time import time

def findAffine(xo, xp):
    Ao = np.row_stack([xo, np.ones(3)])
    Ap = np.row_stack([xp, np.ones(3)])
    M = np.dot(Ap, np.linalg.inv(Ao))
    return M


# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6,5), dpi=1080/5)

t = np.sort(np.random.uniform(0, 1, 20))
x = np.linspace(0, 1, 350)
y = np.linspace(0, 1, 300)
T,X,Y = np.meshgrid(t, y, x, indexing='ij')

Z = np.random.normal(1000, 100, T.shape)

Z += 5_000*np.sin(2*np.pi*T)**2 * np.exp(-((X-0.4)**2 + (Y-0.4)**2)/(2*0.025**2))

plt.imshow(Z[0], origin='lower')

nframes = 10
tinterp = np.linspace(t.min(), t.max(), nframes, endpoint=False)



def update(frame):
    ax.clear()

    ti = tinterp[frame]
    right = np.argmax(t>ti)
    left = right - 1
    # print(t[left], tinterp[frame], t[right])
    im_int = (Z[right]-Z[left])/(t[right]-t[left]) * (ti-t[left]) + Z[left]
    im = ax.imshow(im_int, vmin=0, vmax=6000, interpolation='nearest',
                   extent=(x.min(),x.max(),y.min(),y.max()),
                   origin='lower', rasterized=True)
    # cb = plt.colorbar()
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=nframes)
ani.save('animation.mp4', writer='ffmpeg', fps=10)
plt.close(fig)


tgt = '/Users/bruzewskis/Documents/scripting/workrelated/ifgramload/stripped.h5'
with h5py.File(tgt, 'r') as fo:

    data = fo['coherence'][0]
    data[data==0] = np.nan
    
    ps = np.array([[0, data.shape[1], 0],
                   [0, 0, data.shape[0]]])

    cs = np.array([[float(fo.attrs[f'LON_REF{i+1}']),
                    float(fo.attrs[f'LAT_REF{i+1}']) ] for i in range(4)]).T

    # Hard way
    xext = np.sqrt((cs[0,0]-cs[0,1])**2 + (cs[1,0]-cs[1,1])**2)
    xsc = xext / data.shape[1]
    yext = np.sqrt((cs[0,0]-cs[0,2])**2 + (cs[1,0]-cs[1,2])**2)
    ysc = yext / data.shape[0]
    ang = np.rad2deg(np.arctan2(cs[1,1]-cs[1,0], cs[0,1]-cs[0,0]))
    
    # Easy way
    M = findAffine(ps, cs[:,:3])

    fig, ax = plt.subplots(figsize=(5, 5), dpi=1080/5)
    tr1 = transforms.Affine2D().scale(xsc, ysc).rotate_deg(ang).translate(cs[0,0], cs[1,0])
    tr2 = transforms.Affine2D(M)

    ax.imshow(data, vmin=0, vmax=1, interpolation='nearest',
              origin='lower', rasterized=True, cmap='Spectral_r',
              transform=tr1 + ax.transData)

    r = 0.1
    plt.xlim(np.min(cs[0])-r, np.max(cs[0])+r)
    plt.ylim(np.min(cs[1])-r, np.max(cs[1])+r)
    plt.scatter(*cs, c=np.arange(4))
