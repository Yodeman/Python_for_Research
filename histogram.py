import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(size=1000)     #using the normal distribution
#print(x[:10])
#plt.hist(x)
plt.hist(x, normed=True, bins=np.linspace(-5,5,21))     #normalised histogram
plt.show()