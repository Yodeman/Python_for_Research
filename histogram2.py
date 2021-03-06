import numpy as np
import matplotlib.pyplot as plt

x = np.random.gamma(2, 3, 100000)     #using the gamma distribution
plt.figure()
plt.subplot(221)
plt.hist(x, bins=30)
plt.subplot(222)
plt.hist(x, bins=30, density=True,)
plt.subplot(223)
plt.hist(x, bins=30, cumulative=30)
plt.subplot(224)
plt.hist(x, bins=30, density=True, cumulative=True, histtype="step")
#plt.hist(x, normed=True, bins=np.linspace(-5,5,21))
plt.show()