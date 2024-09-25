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

# Create meshgrid for the entire dataset
X, Y = np.meshgrid(x, y)

def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-((9*x - 2)**2)/4.0 - ((9*y - 2)**2)/4.0)+ np.random.normal(0, 0.01, x.shape)
    term2 = 0.75 * np.exp(-(9*x + 1)**2/49.0 - 0.1*(9*y + 1)) + np.random.normal(0, 0.01, x.shape)
    term3 = 0.5 * np.exp(-(9*x - 7)**2/4.0 - ((9*y - 3)**2)/4.0) + np.random.normal(0, 0.01, x.shape)
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2) + np.random.normal(0, 0.01, x.shape)
    return term1 + term2 + term3 + term4

# Create z values
z = FrankeFunction(X, Y)

# Flattening the X, Y, and z arrays for train/test split
x_flat = X.ravel()
y_flat = Y.ravel()
z_flat = z.ravel()

# Split the data into training and test set (20% test size)
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
    x_flat, y_flat, z_flat, test_size=0.2, random_state=315)

# Maximum polynomial degree
Maxpolydegree = 5

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

# Create design matrices for training and testing
X_design_train = create_design_matrix(x_train, y_train, Maxpolydegree)
X_design_test = create_design_matrix(x_test, y_test, Maxpolydegree)

# Perform OLS regression
beta = inv(X_design_train.T @ X_design_train) @ X_design_train.T @ z_train

# Estimate z values for training and testing data
z_est_train = X_design_train @ beta
z_est_test = X_design_test @ beta



# Reshape back to grid for complete surface plotting, usikker på dennne. 
z_est_grid = create_design_matrix(x_flat, y_flat, Maxpolydegree) @ beta
z_est_grid = z_est_grid.reshape(n, n)

# Plot the original and estimated surfaces
fig = plt.figure(figsize=(18, 6))

# Original Franke Function
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax1.set_title('Original Franke Function')
ax1.set_zlim(-0.10, 1.40)

# Estimated Surface
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, z_est_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax2.set_title('Estimated Surface for Full Data')
ax2.set_zlim(-0.10, 1.40)

# Plot predicted vs actual (training and test)
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(x_train, y_train, z_train, color='blue', label='Training Data')
ax3.scatter(x_train, y_train, z_est_train, color='cyan', label='Training Predictions', marker='x')
ax3.scatter(x_test, y_test, z_test, color='red', label='Test Data')
ax3.scatter(x_test, y_test, z_est_test, color='yellow', label='Test Predictions', marker='x')
ax3.set_title('Training and Test Predictions')
ax3.set_zlim(-0.10, 1.40)
ax3.legend()

plt.show()

# Define MSE and R2 functions
def MSE(z_data, z_model):
    return np.mean((z_data - z_model)**2)

def R2(z_data, z_model):
    return 1 - np.sum((z_data - z_model)**2) / np.sum((z_data - np.mean(z_data))**2)

# Evaluate the model
print("Training MSE:", MSE(z_train, z_est_train))
print("Training R2:", R2(z_train, z_est_train))

print("Test MSE:", MSE(z_test, z_est_test))
print("Test R2:", R2(z_test, z_est_test))

# Loop over different polynomial degrees to check MSE and R2
polydegree = np.arange(1, Maxpolydegree + 1)
Error_train = np.zeros(Maxpolydegree)
Score_train = np.zeros(Maxpolydegree)
Error_test = np.zeros(Maxpolydegree)
Score_test = np.zeros(Maxpolydegree)
beta_compl = []

for degree in range(1, Maxpolydegree + 1):
    X_design_train_temp = create_design_matrix(x_train, y_train, degree)
    X_design_test_temp = create_design_matrix(x_test, y_test, degree)
    
    # Perform OLS regression
    beta_temp = inv(X_design_train_temp.T @ X_design_train_temp) @ X_design_train_temp.T @ z_train
    
    z_est_train_temp = X_design_train_temp @ beta_temp
    z_est_test_temp = X_design_test_temp @ beta_temp
    
    Error_train[degree - 1] = MSE(z_train, z_est_train_temp)
    Score_train[degree - 1] = R2(z_train, z_est_train_temp)
    Error_test[degree - 1] = MSE(z_test, z_est_test_temp)
    Score_test[degree - 1] = R2(z_test, z_est_test_temp)
    beta_compl.append(beta_temp)

# Plot MSE and R2 for different polynomial degrees
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(polydegree, Error_train, label='Train Error')
plt.plot(polydegree, Error_test, label='Test Error')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.legend()
plt.title('MSE vs Polynomial Degree')

plt.subplot(1, 2, 2)
plt.plot(polydegree, Score_train, label='Train R2')
plt.plot(polydegree, Score_test, label='Test R2')
plt.xlabel('Polynomial Degree')
plt.ylabel('R2 Score')
plt.legend()
plt.title('R2 vs Polynomial Degree')

plt.show()

# Plot beta coefficients for different polynomial degrees
plt.figure(figsize=(18, 6))

for i, beta in enumerate(beta_compl, 1):
    plt.bar(range(len(beta)), np.log10(beta), label=f'Beta for degree {i}')
    plt.xticks(np.arange(0, 21, step=1))

plt.xlabel('Coefficient Index')
plt.ylabel('Beta Value')
plt.legend()
plt.title('Beta Coefficients vs Polynomial Degree')
plt.show()
