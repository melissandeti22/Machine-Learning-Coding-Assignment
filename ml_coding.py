import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("../../Downloads/Nairobi Office Price Ex (1).csv")  # Update the path if needed

# Extract relevant columns
X = data['SIZE'].values  # Feature (office size)
y = data['PRICE'].values  # Target (office price)

# Define Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define Gradient Descent function
def gradient_descent(X, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * X + c
    # Calculate gradients
    dm = (-2 / n) * np.sum(X * (y - y_pred))
    dc = (-2 / n) * np.sum(y - y_pred)
    # Update weights
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Initialize slope (m) and intercept (c) randomly
m, c = np.random.rand(), np.random.rand()
learning_rate = 0.0001  # Set learning rate
epochs = 10  # Set number of epochs

# Training loop
for epoch in range(epochs):
    # Predict with current slope and intercept
    y_pred = m * X + c
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y, y_pred)
    print(f"Epoch {epoch + 1}: MSE = {mse:.4f}")
    # Update weights using gradient descent
    m, c = gradient_descent(X, y, m, c, learning_rate)

# Plotting the line of best fit after final epoch
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, m * X + c, color="red", label="Line of Best Fit")
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.legend()
plt.title("Linear Regression - Line of Best Fit")
plt.show()

# Predict the office price for size 100 sq. ft.
predicted_price = m * 100 + c
print(f"Predicted price for a 100 sq. ft. office: {predicted_price:.2f}")
