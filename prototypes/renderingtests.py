import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6,5), dpi=1080/5)

t = np.linspace(0, 1, 100)
x = np.linspace(0, 1, 350)
y = np.linspace(0, 1, 300)
T,X,Y = np.meshgrid(t, y, x, indexing='ij')

Z = np.random.normal(1000, 100, (100, y.size, x.size))

Z += 3000*np.exp(-((X-0.1-0.5*T)**2 + (Y-0.1-0.5*T)**2)/(2*0.025**2))

plt.imshow(Z[0], origin='lower')

def update(frame):
    ax.clear()
    im = ax.imshow(Z[frame], vmin=100, vmax=4000, interpolation='nearest',
                   extent=(x.min(),x.max(),y.min(),y.max()),
                   origin='lower', rasterized=True)
    return (im,)
    
ani = animation.FuncAnimation(fig, update, frames=len(Z))
ani.save('animation.mp4', writer='ffmpeg', fps=10)
plt.close(fig)
