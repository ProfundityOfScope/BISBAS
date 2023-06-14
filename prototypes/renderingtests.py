import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6,5), dpi=1080/5)

t = np.sort(np.random.uniform(0, 1, 20))
x = np.linspace(0, 1, 350)
y = np.linspace(0, 1, 300)
T,X,Y = np.meshgrid(t, y, x, indexing='ij')

Z = np.random.normal(1000, 100, T.shape)

Z += 5_000*np.sin(2*np.pi*T)**2 * np.exp(-((X-0.4)**2 + (Y-0.4)**2)/(2*0.025**2))

plt.imshow(Z[0], origin='lower')

nframes = 100
tinterp = np.linspace(t.min(), t.max(), nframes, endpoint=False)

def update(frame):
    ax.clear()
    
    ti = tinterp[frame]
    right = np.argmax(t>ti)
    left = right - 1
    # print(t[left], tinterp[frame], t[right])
    im_int = (Z[right]-Z[left])/(t[right]-t[left]) * (ti-t[left]) + Z[left]
    print(left, frame, right)
    im = ax.imshow(im_int, vmin=0, vmax=6000, interpolation='nearest',
                   extent=(x.min(),x.max(),y.min(),y.max()),
                   origin='lower', rasterized=True)
    return (im,)
    
ani = animation.FuncAnimation(fig, update, frames=nframes)
ani.save('animation.mp4', writer='ffmpeg', fps=10)
plt.close(fig)