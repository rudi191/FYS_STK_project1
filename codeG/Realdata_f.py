#Solution to f) cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from numpy.linalg import inv
from sklearn.linear_model import Ridge, Lasso
from sklearn.utils import resample
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from imageio import imread
from numpy.linalg import inv
from matplotlib import cm


# Function to create design matrix
def create_design_matrix(x, y, degree):
    num_terms = int((degree + 1)*(degree + 2)/2)
    X = np.zeros((len(x), num_terms))
    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            X[:, idx] = (x**i) * (y**j)
            idx += 1
    return X

# Define MSE and R2 functions
def MSE(z_data, z_model):
    return np.mean((z_data - z_model)**2)

def R2(z_data, z_model):
    return 1 - np.sum((z_data - z_model)**2) / np.sum((z_data - np.mean(z_data))**2)

# Load the terrain data
terrain1 = imread('SRTM_data_Norway_2.tif')
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


# Set up k-fold cross-validation
Maxpolydegree = 10
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=315)

# OLS Regression
ols_mse = np.zeros(Maxpolydegree)

for degree in range(1, Maxpolydegree + 1):
    X_design_temp = create_design_matrix(x_train, y_train, degree)
    model = LinearRegression()
    z_pred = cross_val_predict(model, X_design_temp, z_train, cv=kfold)
    scores = cross_val_score(model, X_design_temp, z_train, scoring='neg_mean_squared_error', cv=kfold)
    ols_mse[degree-1] = -np.mean(scores)

plt.figure()
plt.plot(range(1, Maxpolydegree + 1), ols_mse, marker='o', linestyle='-')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.title('OLS Regression MSE vs Polynomial Degree CV')

plt.show()

# Ridge Regression
nlambdas = 20
lambdas = np.logspace(-4, 4, nlambdas)
estimated_mse_sklearn = np.zeros((Maxpolydegree, nlambdas))

for degree in range(1, Maxpolydegree + 1):
    X_design_temp = create_design_matrix(x_train, y_train, degree)
    for i in range(nlambdas):
        lmbda = lambdas[i]
        ridge = Ridge(alpha=lmbda, fit_intercept=False)
        estimated_mse_folds = cross_val_score(ridge, X_design_temp, z_train, scoring='neg_mean_squared_error', cv=kfold)
        estimated_mse_sklearn[degree-1, i] = np.mean(-estimated_mse_folds)

plt.figure()
for degree in range(1, Maxpolydegree + 1):
    plt.plot(lambdas, estimated_mse_sklearn[degree-1], label=f'Degree {degree}')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.title('Ridge Regression MSE vs Lambda CV')
plt.legend()
plt.show()

# Lasso Regression
estimated_mse_sklearn_lasso = np.zeros((Maxpolydegree, nlambdas))

for degree in range(1, Maxpolydegree + 1):
    X_design_temp = create_design_matrix(x_train, y_train, degree)
    for i in range(nlambdas):
        lmbda = lambdas[i]
        lasso = Lasso(alpha=lmbda, fit_intercept=False, max_iter=10000)
        estimated_mse_folds = cross_val_score(lasso, X_design_temp, z_train, scoring='neg_mean_squared_error', cv=kfold)
        estimated_mse_sklearn_lasso[degree-1, i] = np.mean(-estimated_mse_folds)

plt.figure()
for degree in range(1, Maxpolydegree + 1):
    plt.plot(lambdas, estimated_mse_sklearn_lasso[degree-1], label=f'Degree {degree}')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.title('Lasso Regression MSE vs Lambda CV')
plt.legend()
plt.show()