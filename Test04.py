import numpy as np
values = np.array([10.0,20.0,30.0])
weights = np.full_like(values,1/3)
print("weights:", weights)
print("via mean:", np.mean(values))
print("via weights and dot :", np.dot(weights,values))