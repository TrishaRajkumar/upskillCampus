# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv("traffic.csv")   # place dataset in same folder

# Display Data
print(df.head())

# Convert date column
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extract time features
df['hour'] = df['DateTime'].dt.hour
df['day'] = df['DateTime'].dt.day
df['month'] = df['DateTime'].dt.month

# Drop original column
df = df.drop(['DateTime'], axis=1)

# Handle missing values
df = df.dropna()

# Features & Target
X = df.drop("Vehicles", axis=1)
y = df["Vehicles"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))

# Plot
plt.plot(y_test.values[:50], label="Actual")
plt.plot(y_pred[:50], label="Predicted")
plt.legend()
plt.title("Traffic Prediction")
plt.show()
