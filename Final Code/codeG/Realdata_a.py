import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imageio.v2 import imread
from numpy.linalg import inv
import seaborn as sns
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

# Shuffle and split the data into training and testing sets
idx = np.random.permutation(len(z_flat))  # Randomly shuffle the indices
x_flat, y_flat, z_normalized = x_flat[idx], y_flat[idx], z_normalized[idx]

#as it took a long time to run the whole dataset, we use a subsection.
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
    x_flat, y_flat, z_normalized, test_size=0.02, train_size=0.08, random_state=10003
)

# Centering the data by subtracting the mean
x_mean = np.mean(x_train, axis=0)
y_mean = np.mean(y_train, axis=0)
z_mean = np.mean(z_train, axis=0)

x_train_centered = x_train - x_mean
y_train_centered = y_train - y_mean
z_train_centered = z_train - z_mean

x_test_centered = x_test - x_mean
y_test_centered = y_test - y_mean
z_test_centered = z_test - z_mean

#PART A about the same as Franke function

# Perform OLS and plot MSE and R2 as a function of polynomial degree and the coefficients.
mse_train, mse_test, r2_train, r2_test, coefficients = [], [], [], [], []
degrees = list(range(1, 7))

for poly_degree in degrees:
    XY_train = create_design_matrix(x_train_centered, y_train_centered, poly_degree)
    XY_test = create_design_matrix(x_test_centered, y_test_centered, poly_degree)

    beta = inv(XY_train.T @ XY_train) @ XY_train.T @ z_train_centered

    z_train_pred = XY_train @ beta
    z_test_pred = XY_test @ beta

    mse_train.append(MSE(z_train_centered, z_train_pred))
    mse_test.append(MSE(z_test_centered, z_test_pred))
    r2_train.append(R2(z_train_centered, z_train_pred))
    r2_test.append(R2(z_test_centered, z_test_pred))
    coefficients.append(beta)

#Print the best results
print(f'MSE Train: {min(mse_train)}')
print(f'MSE Test: {min(mse_test)}')
print(f'R2 Train: {max(r2_train)}')
print(f'R2 Test: {max(r2_test)}')

# Print results
print(f'MSE Train: {mse_train}')
print(f'MSE Test: {mse_test}')
print(f'R2 Train: {r2_train}')
print(f'R2 Test: {r2_test}')

# Plotting MSE and R² in one plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plotting MSE
ax1.plot(degrees, mse_train, 'b-', label='Train MSE')
ax1.plot(degrees, mse_test, 'b--', label='Test MSE')
ax1.set_xlabel('Polynomial Degree')
ax1.set_ylabel('Mean Squared Error', color='b')
ax1.tick_params(axis='y', labelcolor='b')
# Creating a second y-axis for R²
ax2 = ax1.twinx()
ax2.plot(degrees, r2_train, 'r-', label='Train R²')
ax2.plot(degrees, r2_test, 'r--', label='Test R²')
ax2.set_ylabel('R² Score', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('MSE and R² vs. Polynomial Degree on normalized data')
fig.tight_layout()
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
plt.savefig("figures\g\MSE_R2_OLSTerrain.png")
plt.show()


# Plotting Coefficients
plt.figure(figsize=(12, 6))
for i, coef in enumerate(coefficients):
    plt.plot(np.arange(len(coef)), coef.flatten(), '-', label=f'Degree {degrees[i]}')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.title('Coefficients vs. Polynomial Degree on normalized data')
plt.legend()
plt.savefig("figures\g\coefficientsOLSTerrain.png")
plt.show()


# Convert coefficients to a 2D matrix with padding as required
def pad_coefficients(coefficients):
    max_length = max(len(coef) for coef in coefficients)
    padded_coefficients = np.array([np.pad(coef, (0, max_length - len(coef)), 'constant') for coef in coefficients])
    return padded_coefficients

coefficients_matrix = pad_coefficients(coefficients)

# Plotting Coefficients as Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(coefficients_matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=np.arange(coefficients_matrix.shape[1]), yticklabels=degrees)
plt.xlabel('Coefficient Index')
plt.ylabel('Polynomial Degree')
plt.title('Coefficients vs. Polynomial Degree on normalized data')
plt.savefig("figures\g\CoefficientsHeatmapOLSTerrain.png")
plt.show()
