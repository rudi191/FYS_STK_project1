import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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
import seaborn as sns


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
    x_flat, y_flat, z_normalized, test_size=0.02, train_size=0.08, random_state=1
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

# Ridge Parameters
Maxpolydegree = 10
nlambdas = 5
lambdas = np.logspace(-4, 0, nlambdas)

# Initialize result storage
Error_train_ridge = np.zeros((Maxpolydegree, nlambdas))
Score_train_ridge = np.zeros((Maxpolydegree, nlambdas))
Error_test_ridge = np.zeros((Maxpolydegree, nlambdas))
Score_test_ridge = np.zeros((Maxpolydegree, nlambdas))
num_coefficients = int((Maxpolydegree + 1)*(Maxpolydegree + 2)/2)
betas_ridge = np.zeros((Maxpolydegree, nlambdas, num_coefficients))
testmatrix2 = create_design_matrix(x_train, y_train, 3)
print("design matrix without scaling",testmatrix2)
for degree in range(1, Maxpolydegree + 1):
    X_design_train = create_design_matrix(x_train, y_train, degree)
    X_design_test = create_design_matrix(x_test, y_test, degree)
    
    for i, lmbda in enumerate(lambdas):
        beta_ridge = inv(X_design_train.T @ X_design_train + lmbda * np.eye(X_design_train.shape[1])) @ X_design_train.T @ z_train
        z_est_train_ridge = X_design_train @ beta_ridge
        z_est_test_ridge = X_design_test @ beta_ridge
        
        Error_train_ridge[degree-1, i] = MSE(z_train, z_est_train_ridge)
        Score_train_ridge[degree-1, i] = R2(z_train, z_est_train_ridge)
        Error_test_ridge[degree-1, i] = MSE(z_test, z_est_test_ridge)
        Score_test_ridge[degree-1, i] = R2(z_test, z_est_test_ridge)
        
        betas_ridge[degree-1, i, :beta_ridge.shape[0]] = beta_ridge
             # Print results best results
        if degree == 10:
            print(f'minimum MSE Train: {np.min(Error_train_ridge)}')
            print(f'minimum MSE Test: {np.min(Error_test_ridge)}')
            print(f'maximum R2 Train: {np.max(Score_train_ridge)}')
            print(f'maximum R2 Test: {np.max(Score_test_ridge)}')

# Plotting MSE for Ridge Regression
plt.figure(figsize=(12, 6))
for i, lmbda in enumerate(lambdas):
    plt.plot(range(1, Maxpolydegree + 1), Error_train_ridge[:, i], '-', label=f'Train MSE, Lambda={lmbda:.4f}')
    plt.plot(range(1, Maxpolydegree + 1), Error_test_ridge[:, i], '--', label=f'Test MSE, Lambda={lmbda:.4f}')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Polynomial Degree for different Lambda values (Ridge Regression)')
plt.legend()
plt.savefig("figures/MSERidgeTerrain.png")
plt.show()

# Plotting R² for Ridge Regression
plt.figure(figsize=(12, 6))
for i, lmbda in enumerate(lambdas):
    plt.plot(range(1, Maxpolydegree + 1), Score_train_ridge[:, i], '-', label=f'Train R², Lambda={lmbda:.4f}')
    plt.plot(range(1, Maxpolydegree + 1), Score_test_ridge[:, i], '--', label=f'Test R², Lambda={lmbda:.4f}')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('R² vs. Polynomial Degree for different Lambda values (Ridge Regression)')
plt.legend()
plt.savefig("figures/R2RidgeTerrain.png")
plt.show()

# Plotting R² for Ridge Regression as Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(Error_test_ridge, annot=True, fmt=".2f", xticklabels=np.round(lambdas, 4), yticklabels=range(1, Maxpolydegree + 1), cmap="viridis")
plt.xlabel('Lambda')
plt.ylabel('Polynomial Degree')
plt.title('Test MSE Heatmap for Ridge Regression')
plt.savefig("figures/TrainR2HeatmapRidgeTerrain.png")
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(Error_train_ridge, annot=True, fmt=".2f", xticklabels=np.round(lambdas, 4), yticklabels=range(1, Maxpolydegree + 1), cmap="viridis")
plt.xlabel('Lambda')
plt.ylabel('Polynomial Degree')
plt.title('Train MSE Heatmap for Ridge Regression')
plt.savefig("figures/TestR2HeatmapRidgeTerrain.png")
plt.show()