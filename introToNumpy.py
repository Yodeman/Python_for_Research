import numpy as np

# zero arrays
#np.ones for arrays of ones.
zero_vector = np.zeros(5)
zero_matrix = np.zeros((5,3))

#non-zero arrays
x = np.array([1,2,3])
y = np.array([2,4,6])

#multidimensional array
multi_dimen = np.array([[1,2], [3,4]])
#transpose of array
multi_dimen.transpose()

"""slicing numpy arrays"""
x = np.array([1,2,3])
y = np.array([2,4,6])
X = np.array([[1,2,3], [4,5,6]])
Y = np.array([[2,4,6], [8,10,12]])

x[2]
x[0:2]

z = x + y

X[:,1]  #returns the elements in column 1 of X
X[1,:]  #returns the element in row 1 of X

"""indexing numpy arrays"""
z1 = np.array([1,3,5,7,9])
z2 = z1 + 1     # adds 1 to all element in z1
ind = [0,2,3]   # or ind = np.array([0,2,3])
z1[ind]     #indexing numpy array with element in a list
z1[z1 > 6]  #returns array of element in z1 greater than 6

"""linearl spaced array"""
z = np.linspace(0, 100, 10)
"""logarithmcally spaced array"""
z = np.logspace(1,2,10)
z = np.logspace(np.log10(250), np.log10(500), 10)
"""shape, size of array"""
z = np.array([1,2,3], [4,5,6])
z.shape
z.size
"""generating random elem"""
z = np.random.random(10)    # generate 10 random number between 0 and 1