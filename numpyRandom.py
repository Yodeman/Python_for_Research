# using numpy for die.py

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

start = perf_counter()
X = np.random.randint(1, 7, (1000000, 10))
Y = np.sum(X, axis=1)
print("exhausted time is %s" %(perf_counter()-start))
plt.hist(Y)
plt.show()