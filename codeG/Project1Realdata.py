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



# Load the terrain data
terrain1 = imread('SRTM_data_Norway_2.tif')
n_rows, n_cols = terrain1.shape
print("dimesnions of data", n_rows, n_cols)

# Create linearly spaced values
x = np.linspace(0, 1, n_cols)
y = np.linspace(0, 1, n_rows)


# Create meshgrid for the entire dataset
X, Y = np.meshgrid(x, y)
x_flat = X.flatten()
y_flat = Y.flatten()

# Normalize z values to the range [0, 1]
z_flat = terrain1.flatten()
z_min = z_flat.min()
z_max = z_flat.max()
z_normalized = (z_flat - z_min) / (z_max - z_min)

#plotting the terrain data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, terrain1, cmap=cm.viridis)
ax.set_title('Terrain data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig("figures\g\Terrain_data.png")
plt.show()

#plotting 2D image of the terrain data
plt.figure()
plt.imshow(terrain1, cmap='Accent')
plt.title('Terrain data')
plt.colorbar(label='Height')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("figures\g\Terrain_data2D.png")
plt.show()




# Split the data into training and test set (20% test size)
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
    x_flat, y_flat, z_normalized, train_size=0.08,test_size=0.02,  random_state=315)

print("shape of x_train", x_train.shape)
print("shape of x_test", x_test.shape)

# Maximum polynomial degree
Maxpolydegree = 6

def create_design_matrix(x, y, degree):
    num_terms = int((degree + 1)*(degree + 2)/2)
    X = np.zeros((len(x), num_terms))
    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            X[:, idx] = (x**i) * (y**j)
            idx += 1
    return X

# Function to calculate MSE
def MSE(z_data, z_model):
    return np.mean((z_data - z_model)**2)

# Function to calculate R2
def R2(z_data, z_model):
    return 1 - np.sum((z_data - z_model)**2) / np.sum((z_data - np.mean(z_data))**2)



X_design_train = create_design_matrix(x_train, y_train, Maxpolydegree)
X_design_test = create_design_matrix(x_test, y_test, Maxpolydegree)

beta = inv(X_design_train.T @ X_design_train) @ X_design_train.T @ z_train
z_est_train = X_design_train @ beta
z_est_test = X_design_test @ beta

# Reshape back to grid for complete surface plotting
z_est_grid = create_design_matrix(x_flat, y_flat, Maxpolydegree) @ beta
z_est_grid = z_est_grid.reshape(n_rows, n_cols)

# Plot the original and estimated surfaces
fig = plt.figure(figsize=(18, 6))

# Original Terrain Data
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, terrain1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax1.set_title('Original Terrain Data')

# Estimated Surface
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, z_est_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax2.set_title('Estimated Surface for Full Data')
plt.savefig("figures\g\Terrain_datavsFit.png")
plt.show()

# Calculate MSE and R2 for training and test data
MSE_train = MSE(z_train, z_est_train)
MSE_test = MSE(z_test, z_est_test)
R2_train = R2(z_train, z_est_train)
R2_test = R2(z_test, z_est_test)

print(f'MSE for training data: {MSE_train:.4f}')
print(f'MSE for test data: {MSE_test:.4f}')
print(f'R2 for training data: {R2_train:.4f}')

print(f'R2 for test data: {R2_test:.4f}')



