#Solution to f) cross-validation
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso
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

# Create meshgrid for the entire dataset
X, Y = np.meshgrid(x, y)
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = terrain1.flatten()

# Normalize the z values
z_min = z_flat.min()
z_max = z_flat.max()
z_normalized = (z_flat - z_min) / (z_max - z_min)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
    x_flat, y_flat, z_normalized, test_size=0.002, train_size=0.008, random_state=315
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

#PART F about the same as Franke function

# Set up k-fold cross-validation
Maxpolydegree = 10
k = 10
kfold = KFold(n_splits=k, shuffle=True, random_state=315)

# OLS Regression
ols_mse = np.zeros(Maxpolydegree)

for degree in range(1, Maxpolydegree + 1):
    X_design_temp = create_design_matrix(x_train, y_train, degree)
    model = LinearRegression()
    z_pred = cross_val_predict(model, X_design_temp, z_train, cv=kfold)
    scores = cross_val_score(model, X_design_temp, z_train, scoring='neg_mean_squared_error', cv=kfold)
    ols_mse[degree-1] = -np.mean(scores)

# Ridge Regression
nlambdas = 5
lambdas = np.logspace(-5,-2 , nlambdas)
ridge_mse = np.zeros((Maxpolydegree, nlambdas))

for degree in range(1, Maxpolydegree + 1):
    X_design_temp = create_design_matrix(x_train, y_train, degree)
    for i in range(nlambdas):
        lmbda = lambdas[i]
        ridge = Ridge(alpha=lmbda, fit_intercept=False)
        estimated_mse_folds = cross_val_score(ridge, X_design_temp, z_train, scoring='neg_mean_squared_error', cv=kfold)
        ridge_mse[degree-1, i] = np.mean(-estimated_mse_folds)

# Lasso Regression
lasso_mse = np.zeros((Maxpolydegree, nlambdas))

for degree in range(1, Maxpolydegree + 1):
    X_design_temp = create_design_matrix(x_train, y_train, degree)
    for i in range(nlambdas):
        lmbda = lambdas[i]
        lasso = Lasso(alpha=lmbda, fit_intercept=False, max_iter=10000)
        estimated_mse_folds = cross_val_score(lasso, X_design_temp, z_train, scoring='neg_mean_squared_error', cv=kfold)
        lasso_mse[degree-1, i] = np.mean(-estimated_mse_folds)

# Plotting all three MSEs in one plot
plt.figure(figsize=(12, 8))

# OLS MSE plot
plt.plot(range(1, Maxpolydegree + 1), ols_mse, marker='o', linestyle='-', label='OLS')

# Ridge MSE plot
for i, lmbda in enumerate(lambdas):
    plt.plot(range(1, Maxpolydegree + 1), ridge_mse[:, i], linestyle='--', label=f'Ridge Lambda={lmbda:.4f}')

# Lasso MSE plot
for i, lmbda in enumerate(lambdas):
    plt.plot(range(1, Maxpolydegree + 1), lasso_mse[:, i], linestyle='-', label=f'Lasso Lambda={lmbda:.4f}')

plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.title('MSE vs Polynomial Degree for OLS, Ridge, and Lasso Regression')
plt.legend()
plt.savefig("figures\g\CVoneplotTerrain2.png")
plt.show()
