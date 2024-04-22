import numpy as np
import matplotlib.pyplot as plt
number_people = np.arange(1,11)
number_beer = np.arange(0,20)
number_people,number_beer=np.meshgrid(number_people,number_beer )

total_cost= 80* number_people**2 + 10*number_beer**2 +40

from mpl_toolkits.mplot3d import Axes3D
fig,axes =plt.subplots(2,3,
                       subplot_kw={'projection':'3d'},
                       figsize=(9,6),
                       tight_layout=True)
angles=[0,45,90,135,180]
for ax, angle in zip(axes.flat, angles):
    ax.plot_surface(number_people,number_beer,total_cost)
    ax.set_xlabel('People')
    ax.set_ylabel('Beer')
    ax.set_zlabel('TotalCost')
    ax.azim = angle
axes.flat[-1].axis('off')
fig.tight_layout()
plt.show()




