import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imageio.v2 import imread
from numpy.linalg import inv
import os

# DATA GENERATION - SAME FOR A-F
def create_design_matrix(x, y, degree):
    num_terms = int((degree + 1)*(degree + 2)/2)
    X = np.zeros((len(x), num_terms))
    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            X[:, idx] = (x**i) * (y**j)
            idx += 1
    return X

# Defining MSE and R2 functions
def MSE(z_data, z_ols_model):
    return np.mean((z_data - z_ols_model)**2)

# The R2 function was taken from Week 35: From Ordinary Linear Regression to Ridge and Lasso Regression,
# Morten Hjorth-Jensen, Department of Physics, University of Oslo.
# https://github.com/CompPhysics/MachineLearning/blob/master/doc/LectureNotes/week37.ipynb
def R2(z_data, z_ols_model):
    return 1 - np.sum((z_data - z_ols_model)**2) / np.sum((z_data - np.mean(z_data))**2)

current_dir = os.path.dirname("codeG")
file_path = os.path.join(current_dir, '..', '..', 'Datafiles', 'SRTM_data_Norway_2.tif')
terrain1 = imread(file_path)

n_rows, n_cols = terrain1.shape

# Create linearly spaced values
x = np.linspace(0, 1, n_cols)
y = np.linspace(0, 1, n_rows)

X, Y = np.meshgrid(x, y)
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = terrain1.flatten()

#as x and y is between 0 and 1 and z values were vastly different we need to normalize the values
z_min = z_flat.min()
z_max = z_flat.max()
z_normalized = (z_flat - z_min) / (z_max - z_min)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
    x_flat, y_flat, z_normalized, test_size=0.0002, train_size=0.0008, random_state=315
)
# Centering the data by subtracting the mean, the values are already scaled between 0 and 1. 
x_mean = np.mean(x_train, axis=0)
y_mean = np.mean(y_train, axis=0)
z_mean = np.mean(z_train, axis=0)

x_train = x_train - x_mean
y_train = y_train - y_mean
z_train = z_train - z_mean

x_test = x_test - x_mean
y_test = y_test - y_mean
z_test = z_test - z_mean

#PART E about the same as Franke function
from sklearn.utils import resample

# Bootstrapping unscaled data
Maxpolydegree = 21
n_boostraps = 100


error = np.zeros(Maxpolydegree)
bias = np.zeros(Maxpolydegree)
variance = np.zeros(Maxpolydegree)
polydegree = np.zeros(Maxpolydegree)
#testmatrix4 = create_design_matrix(x_train, y_train, 3)
#print("design matrix without scaling",testmatrix4)

for degree in range(1, Maxpolydegree + 1):
    z_pred = np.empty((z_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_, z_ = resample(x_train, y_train, z_train)
        design_m = create_design_matrix(x_, y_, degree) 
        beta_m = inv(design_m.T @ design_m) @ design_m.T @ z_
        z_pred[:, i] = create_design_matrix(x_test, y_test, degree) @ beta_m

    polydegree[degree-1] = degree
    error[degree-1] = np.mean((z_test[:, np.newaxis] - z_pred)**2)
    bias[degree-1] = np.mean((z_test[:, np.newaxis] - np.mean(z_pred, axis=1, keepdims=True))**2)
    variance[degree-1] = np.mean(np.var(z_pred, axis=1, keepdims=True))

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='Bias')
plt.plot(polydegree, variance, label='Variance')
plt.title("Bias, Error, and Variance for OLS on Terrain Data")
plt.xlabel('Polynomial Degree')
plt.ylabel('Score')
plt.legend()
plt.savefig("figures\g\BiasVarOLSTerrain.png")
plt.show()