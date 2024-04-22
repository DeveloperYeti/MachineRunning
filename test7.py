from mlwpy import *
import numpy as np
import matplotlib.pyplot as plt
xs = np.linspace(-3,3,100)
m,b=1.5, -3
ax = plt.gca()

ys =  m * xs +b
ax.plot(xs,ys)

ax.set_ylim(-4,4)
high_school_style(ax)

ax.plot(0,-3,'ro')
ax.plot(2,0,'ro')
ys= 0* xs+b
ax.plot(xs,ys,'y')
plt.show()