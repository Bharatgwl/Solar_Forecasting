import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = pd.read_csv("solarpowergeneration.csv")  # Replace with your dataset

# Split features and target
X = data.iloc[:, :-1].values  # All columns except last as features
y = data.iloc[:, -1].values   # Last column as target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# Save the scalers
joblib.dump(sc_X, 'scaler_X.pkl')
joblib.dump(sc_y, 'scaler_y.pkl')

# Build ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, sc_y.transform(y_test.reshape(-1, 1))))

# Save the trained model
model.save('solar_power_forecasting_model.h5')

print("Model and scalers saved successfully!")

# To load the model later
# from tensorflow.keras.models import load_model
# loaded_model = load_model('solar_power_forecasting_model.h5')
# sc_X = joblib.load('scaler_X.pkl')
# sc_y = joblib.load('scaler_y.pkl')
