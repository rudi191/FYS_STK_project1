import numpy as np
from sklearn.model_selection import train_test_split
from imageio.v2 import imread
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
    x_flat, y_flat, z_normalized, test_size=0.002,train_size=0.008, random_state=315
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

#PART C about the same as Franke function
from sklearn.linear_model import Lasso
import numpy as np
import matplotlib.pyplot as plt

# Define constants and initialize containers
Maxpolydegree = 10
nlambdas = 5
lambdas = np.logspace(-4, 0, nlambdas)

Error_train_lasso = np.zeros((Maxpolydegree, nlambdas))
Score_train_lasso = np.zeros((Maxpolydegree, nlambdas))
Error_test_lasso = np.zeros((Maxpolydegree, nlambdas))
Score_test_lasso = np.zeros((Maxpolydegree, nlambdas))

num_coefficients = int((Maxpolydegree + 1)*(Maxpolydegree + 2)/2)
betas_lasso = np.zeros((Maxpolydegree, nlambdas, num_coefficients))

#testmatrix3 = create_design_matrix(x_train, y_train, 3)
#print("design matrix without scaling",testmatrix3)

for degree in range(1, Maxpolydegree + 1):
    X_design_train = create_design_matrix(x_train, y_train, degree)
    X_design_test = create_design_matrix(x_test, y_test, degree)

    for i, lmbda in enumerate(lambdas):
        lasso = Lasso(alpha=lmbda, fit_intercept=False, max_iter=10000)
        lasso.fit(X_design_train, z_train)
        beta_lasso = lasso.coef_
        
        z_est_train_lasso = lasso.predict(X_design_train)
        z_est_test_lasso = lasso.predict(X_design_test)
        
        Error_train_lasso[degree-1, i] = MSE(z_train, z_est_train_lasso)
        Score_train_lasso[degree-1, i] = R2(z_train, z_est_train_lasso)
        Error_test_lasso[degree-1, i] = MSE(z_test, z_est_test_lasso)
        Score_test_lasso[degree-1, i] = R2(z_test, z_est_test_lasso)
        
        betas_lasso[degree-1, i, :beta_lasso.shape[0]] = beta_lasso

# Print results best results
print(f'minimum MSE Train: {np.min(Error_train_lasso)}')
print(f'minimum MSE Test: {np.min(Error_test_lasso)}')
print(f'maximum R2 Train: {np.max(Score_train_lasso)}')
print(f'maximum R2 Test: {np.max(Score_test_lasso)}')

# Plotting MSE for LASSO
plt.figure(figsize=(12, 6))
for i, lmbda in enumerate(lambdas):
    plt.plot(range(1, Maxpolydegree + 1), Error_train_lasso[:, i], '-', label=f'Train MSE, Lambda={lmbda:.4f}')
    plt.plot(range(1, Maxpolydegree + 1), Error_test_lasso[:, i], '--', label=f'Test MSE, Lambda={lmbda:.4f}')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Polynomial Degree for different Lambda values (Lasso Regression)')
plt.legend()
plt.savefig("figures\g\MSELassoTerrain.png")
plt.show()

# Plotting R² for LASSO
plt.figure(figsize=(12, 6))
for i, lmbda in enumerate(lambdas):
    plt.plot(range(1, Maxpolydegree + 1), Score_train_lasso[:, i], '-', label=f'Train R², Lambda={lmbda:.4f}')
    plt.plot(range(1, Maxpolydegree + 1), Score_test_lasso[:, i], '--', label=f'Test R², Lambda={lmbda:.4f}')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('R² vs. Polynomial Degree for different Lambda values (Lasso Regression)')
plt.legend()
plt.savefig("figures\g\R2LassoTerrain.png")
plt.show()