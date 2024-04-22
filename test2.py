import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

b =ss.distributions.binom
for flips in [5,10,20,40,80]:
    success = np.arange(flips)
    out_distribution = b.pmf(success,flips,0.5)
    plt.hist(success,flips,weights=out_distribution)
plt.xlim(0,55)
plt.show()

