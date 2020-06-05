import matplotlib.pyplot as plt
import numpy as np

#plt.plot([0,1,4,9,16])
#plt.show()

x = np.logspace(-1,1,40)
y1 = x ** 2.0
y2 = x**1.5
plt.loglog(x, y1, "bo-", linewidth=2, markersize=4, label="First")    #label == tag
plt.loglog(x, y2, "gs-", linewidth=2, markersize=4, label="Second")

plt.xlabel("X")     #set label for x axis
plt.ylabel("Y")     #set label for y axis
plt.axis([-0.5, 10.5, -5, 105])
plt.legend(loc="upper left")        #show the tag for each graph at the inputed location
plt.savefig("plot2.pdf")
#plt.show()