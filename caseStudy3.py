import numpy as np
import random
import pandas as pd
import sklearn.preprocessing as skp
import sklearn.decomposition as skd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import KNeighborsClassifier
import PythonForResearch.py.classification as hmc

np.random.seed(1)

data = pd.DataFrame(pd.read_csv("wine.csv"))

def sub_dframe(dframe):
    sub_dataFrame = pd.DataFrame(dframe)
    sub_dataFrame.loc[dframe["color"]=="red", "is_red"] = 1
    sub_dataFrame.loc[dframe["color"]=="white", "is_red"]=0
    sub_dataFrame = sub_dataFrame.drop(["color","high_quality", "quality"], axis=1)
    sub_dataFrame["is_red"] = sub_dataFrame["is_red"].astype(int)
    
    return sub_dataFrame

#data["is_red"] = (data["color"]=="red").astype(int)
#numeric_data = data.drop(["color", "high_quality", "quality"], axis=1)
numeric_data = sub_dframe(data)
#print(numeric_data.groupby("is_red").count())

scaled_data = skp.scale(numeric_data)
numeric_data = pd.DataFrame(scaled_data, columns=numeric_data.columns)
#pca = skd.PCA(n_components=2)
#principal_components = pca.fit_transform(numeric_data)
#print(principal_components.shape)

"""observation_colormap = ListedColormap(["red", "blue"])
x = principal_components[:,0]
y = principal_components[:,1]

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha=0.2, c=data["high_quality"],
            cmap=observation_colormap, edgecolors="none"
        )
plt.xlim(-8, 8);plt.ylim(-8, 8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()"""

#x = np.random.randint(0, 2, 1000)
#y = np.random.randint(0, 2, 1000)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(numeric_data, data["high_quality"])
library_predictions = knn.predict(numeric_data)

def accuracy(predictions, outcomes):
    return (100*np.mean(predictions==outcomes))

#print(accuracy(library_predictions, data["high_quality"]))

#Exercise 8
n_rows = data.shape[0]
random.seed(123)
selection = random.sample(range(n_rows), 10)
#print(selection)

#Exercise 9
predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
outcomes = np.array(data["high_quality"])



my_prediction = np.array([hmc.kNN_predict(p, predictors[training_indices,:],
                            outcomes[training_indices], 5) for p in predictors[selection]])
#print(my_prediction[:10])
percentage = accuracy(my_prediction, data.high_quality.iloc[selection])
print(percentage)