# simulates the rolling of a die and plotting the number of outcomes

import random
import matplotlib.pyplot as plt
from time import perf_counter
#import numpy as np

ys = []
start = perf_counter()
for i in range(1000000):
    #throw 10 dice 1000000 times
    y = 0
    for k in range(10):
        #throw 10 dice
        x = random.choice([1,2,3,4,5,6])
        y += x      # add the outcome of the 10 dice
    ys.append(y)
print("exhausted time is %s" %(perf_counter()-start))

plt.hist(ys)
plt.show()