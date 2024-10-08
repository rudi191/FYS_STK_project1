

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import inv, LinAlgError
import numpy as np
from sklearn.linear_model import Ridge

# This code is a verification of the manual implementation of OLS and Ridge regression applied on the Franke Function
# It compares the manual implementation with the scikit-learn implementation
# This includes the creation of the design matrix, calculation of the beta coefficients, and MSE and R2 score calculation

# Defining a function for creating the design matrix
def create_design_matrix(x, y, degree):
    num_terms = int((degree + 1) * (degree + 2) / 2)
    X = np.zeros((len(x), num_terms))
    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            X[:, idx] = (x ** i) * (y ** j)
            idx += 1
    return X

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def R2(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

# Example of a simple manually created dataset
n = 5
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X = np.column_stack((x, y))
print("Simple Dataset", X)

# Function to create polynomial features with scikit-learn
def sklearn_poly_features(X, degree, include_bias=True):
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    return poly.fit_transform(X)

# Verifying the design matrix creation
degrees = [1]
for poly_degree in degrees:
    # Manual
    X_manual = create_design_matrix(x, y, poly_degree)
    print("Manual Design Matrix Degree", poly_degree, X_manual)

    # Scikit-learn
    X_sklearn = sklearn_poly_features(X, poly_degree)
    print("Scikit-learn Design Matrix Degree", poly_degree, X_sklearn)

# Define the Franke Function
def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-((9 * x - 2) ** 2) / 4.0 - ((9 * y - 2) ** 2) / 4.0)
    term2 = 0.75 * np.exp(-(9 * x + 1) ** 2 / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - ((9 * y - 3) ** 2) / 4.0)
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4 + 0.1 * np.random.randn(*x.shape)

# Number of data points
n = 20
np.random.seed(42)

# Create linearly spaced values and meshgrid
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)
x_flat = X.ravel()
y_flat = Y.ravel()
z = FrankeFunction(X, Y)
z_flat = z.ravel()
# Prepare data for sklearn PolynomialFeatures
XY = np.column_stack((x_flat, y_flat))

# Degree of polynomial
degree = 5

# Sci-kit learn OLS implementation
poly = PolynomialFeatures(degree=degree, include_bias=False)
design = poly.fit_transform(XY)
scikitOLS = LinearRegression()
scikitOLS.fit(design, z_flat)
z_pred_sklearn = scikitOLS.predict(design)
print("MSE OLS scikit:", mean_squared_error(z_flat, z_pred_sklearn))
print("R2 OLS scikit:", r2_score(z_flat, z_pred_sklearn))

# Manual OLS implementation

X_design_manual = create_design_matrix(x_flat, y_flat, degree)
beta_manual = inv(X_design_manual.T @ X_design_manual) @ X_design_manual.T @ z_flat
z_model_manual = X_design_manual @ beta_manual
print("MSE OLS manual:", MSE(z_flat, z_model_manual))
print("R2 OLS manual:", R2(z_flat, z_model_manual))



# Ridge regression with scikit-learn
ridge = Ridge(alpha=1.0)
ridge.fit(design, z_flat)
z_pred_ridge = ridge.predict(design)
print("MSE Ridge scikit:", mean_squared_error(z_flat, z_pred_ridge))
print("R2 Ridge scikit:", r2_score(z_flat, z_pred_ridge))

# Ridge manual implementation
lmbda = 1.0
X_design_manual = create_design_matrix(x_flat, y_flat, degree)
beta_ridge_manual = inv(X_design_manual.T @ X_design_manual + lmbda*np.eye(X_design_manual.shape[1])) @ X_design_manual.T @ z_flat
z_model_ridge_manual = X_design_manual @ beta_ridge_manual
print("MSE Ridge manual:", MSE(z_flat, z_model_ridge_manual))
print("R2 Ridge manual:", R2(z_flat, z_model_ridge_manual))


 