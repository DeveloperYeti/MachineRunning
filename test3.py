
import numpy as np
#
# quantity =[2,12,3]
# costs = [12.5 , .5, 1.75]
# partical_cost =[]
# for q, c in zip(quantity,costs):
#     partical_cost.append(q*c)
#     print(sum(partical_cost))
# quantity = [2,12,3]
# costs=[12.5,0.5,1.75]
# print(np.array(quantity)*np.array(costs))
# print(sum(np.array(quantity)*np.array(costs)))

quantity = np.array([2,12,3])
costs = np.array([12.5,0.5,1.75])
print(quantity.dot(costs))
print(np.dot(quantity,costs))
print(quantity@ costs)