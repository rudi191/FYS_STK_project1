import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from matplotlib import cm

# Set random seed for reproducibility
np.random.seed(315)

# Number of data points
n = 20

# Create linearly spaced values
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)

# Create meshgrid
X, Y = np.meshgrid(x, y)

def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-((9*x - 2)**2)/4.0 - ((9*y - 2)**2)/4.0)
    term2 = 0.75 * np.exp(-(9*x + 1)**2/49.0 - 0.1*(9*y + 1))
    term3 = 0.5 * np.exp(-(9*x - 7)**2/4.0 - ((9*y - 3)**2)/4.0)
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(X, Y).ravel()

# Maximum polynomial degree
Maxpolydegree = 5

# Function to create design matrix
def create_design_matrix(x, y, degree):
    num_terms = int((degree + 1)*(degree + 2)/2)  # Number of polynomial terms
    X = np.zeros((len(x), num_terms))
    idx = 0
    for i in range(degree+1):
        for j in range(degree+1 - i):
            X[:, idx] = (x**i) * (y**j)
            idx += 1
    return X

# Create design matrix
x_flat = X.ravel()
y_flat = Y.ravel()
X_design = create_design_matrix(x_flat, y_flat, Maxpolydegree)

# Splitting data
X_train, X_test, z_train, z_test = train_test_split(X_design, z, test_size=0.2)

# Perform OLS regression
beta = inv(X_train.T @ X_train) @ X_train.T @ z_train

# Estimate z values
z_est_train = X_train @ beta
z_est_grid = (X_design @ beta).reshape(n, n)

# Plot the original and estimated surfaces
fig = plt.figure(figsize=(12, 6))

# Original Franke Function
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, FrankeFunction(X, Y), cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax1.set_title('Original Franke Function')
ax1.set_zlim(-0.10, 1.40)
fig.colorbar(surf1, shrink=0.5, aspect=5, ax=ax1)

# Estimated Surface
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, z_est_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax2.set_title('Estimated Surface')
ax2.set_zlim(-0.10, 1.40)
fig.colorbar(surf2, shrink=0.5, aspect=5, ax=ax2)

plt.show()

# Making the MSE and R2 functions
def MSE(z_data, z_model):
    return np.mean((z_data - z_model)**2)

def R2(z_data, z_model):
    return 1 - np.sum((z_data - z_model)**2) / np.sum((z_data - np.mean(z_data))**2)

# Calculate MSE and R2
z_est_test = X_test @ beta
print("The MSE is", MSE(z_test, z_est_test), "The R2 is", R2(z_test, z_est_test))

# Loop over different polynomial degrees to check MSE and R2
polydegree = np.arange(1, Maxpolydegree + 1)
Error = np.zeros(Maxpolydegree)
Score = np.zeros(Maxpolydegree)
betas = np.zeros(Maxpolydegree)

for degree in range(1, Maxpolydegree + 1):
    X_design_temp = create_design_matrix(x_flat, y_flat, degree)
    X_train_temp, X_test_temp, z_train_temp, z_test_temp = train_test_split(X_design_temp, z, test_size=0.2)
    
    # Perform OLS regression
    beta_temp = inv(X_train_temp.T @ X_train_temp) @ X_train_temp.T @ z_train_temp
    
    z_est_temp = X_design_temp @ beta_temp
    betas[degree - 1] = np.mean(beta_temp)
    Error[degree - 1] = MSE(z, z_est_temp)
    Score[degree - 1] = R2(z, z_est_temp)

print("The MSE values are", Error)
plt.plot(polydegree, Error, label='Error')
plt.legend()
plt.show()
