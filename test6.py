import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.display.max_columns = None
people = np. arange(1,11)
total_cost = 80.0 * people + 40.0
print(pd.DataFrame(
    {'total_cost': total_cost},
    index=people).T
)

ax = plt.gca()
ax.plot(people,total_cost,'bo')
ax.set_xlabel("#People")
ax.set_ylabel("#Cost")
plt.show()
