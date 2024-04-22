from mlwpy import *
import numpy as np
import matplotlib.pyplot as plt
xs =np.linspace(-3,3,10)
xs_p1 = np.c_[xs,np.ones_like(xs)]
w=np.array([1.5,-3])
ys = np.dot(xs_p1,w)
ax=plt.gca()
ax.plot(xs,ys)
ax.set_ylim(-4,4)
high_y=np.max(ax)
ax.plot(0,-3,'ro')
ax.plot(2,0,'ro')
plt.show()