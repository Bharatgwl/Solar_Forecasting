# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import joblib
# from tensorflow.keras.models import load_model
# import tensorflow.keras.losses
# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend requests

# # Load the trained model and scalers
# try:
#     # model = load_model("solar_power_forecasting_model.h5")  # Load ANN model
#     model = load_model("solar_power_forecasting_model.h5", custom_objects=custom_objects)
#     sc_X = joblib.load("scaler_X.pkl")  # Load input scaler
#     sc_y = joblib.load("scaler_y.pkl")  # Load target scaler
#     print("✅ Model and scalers loaded successfully.")
# except Exception as e:
#     print(f"❌ Error loading model/scalers: {str(e)}")
#     model, sc_X, sc_y = None, None, None

# # API route to predict solar power generation
# @app.route("/predict", methods=["POST"])
# def predict():
#     if not model or not sc_X or not sc_y:
#         return jsonify({"error": "Model or scalers not loaded properly!"})

#     try:
#         # Define the expected input features
#         input_features = [
#             "temperature_2_m_above_gnd",
#             "relative_humidity_2_m_above_gnd",
#             "mean_sea_level_pressure_MSL",
#             "total_precipitation_sfc",
#             "snowfall_amount_sfc",
#             "total_cloud_cover_sfc",
#             "high_cloud_cover_high_cld_lay",
#             "medium_cloud_cover_mid_cld_lay",
#             "low_cloud_cover_low_cld_lay",
#             "shortwave_radiation_backwards_sfc",
#             "wind_speed_10_m_above_gnd",
#             "wind_direction_10_m_above_gnd",
#             "wind_speed_80_m_above_gnd",
#             "wind_direction_80_m_above_gnd",
#             "wind_speed_900_mb",
#             "wind_direction_900_mb",
#             "wind_gust_10_m_above_gnd",
#             "angle_of_incidence",
#             "zenith",
#             "azimuth",
#         ]

#         # Receive data from frontend
#         data = request.json

#         # Validate input data
#         if not all(feature in data for feature in input_features):
#             return jsonify({"error": "Missing one or more required input fields!"})

#         # Convert input data to numpy array and apply standard scaling
#         input_data = np.array([data[feature] for feature in input_features]).reshape(1, -1)
#         input_scaled = sc_X.transform(input_data)

#         # Predict using ANN model
#         prediction_scaled = model.predict(input_scaled)
#         prediction = sc_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]

#         return jsonify({"predicted_solar_power": float(prediction)})

#     except Exception as e:
#         return jsonify({"error": str(e)})

# # Run Flask server
# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.losses

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define custom objects for model loading
custom_objects = {"mse": tensorflow.keras.losses.MeanSquaredError()}

# Load trained model and scalers
try:
    model = load_model("solar_power_forecasting_model.h5", custom_objects=custom_objects)
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    print("✅ Model and scalers loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/scalers: {str(e)}")
    model, scaler_X, scaler_y = None, None, None

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or scaler_X is None or scaler_y is None:
            return jsonify({"error": "Model or scalers not loaded properly."})

        # Define expected input features
        input_col = [
            "temperature_2_m_above_gnd",
            "relative_humidity_2_m_above_gnd",
            "mean_sea_level_pressure_MSL",
            "total_precipitation_sfc",
            "snowfall_amount_sfc",
            "total_cloud_cover_sfc",
            "high_cloud_cover_high_cld_lay",
            "medium_cloud_cover_mid_cld_lay",
            "low_cloud_cover_low_cld_lay",
            "shortwave_radiation_backwards_sfc",
            "wind_speed_10_m_above_gnd",
            "wind_direction_10_m_above_gnd",
            "wind_speed_80_m_above_gnd",
            "wind_direction_80_m_above_gnd",
            "wind_speed_900_mb",
            "wind_direction_900_mb",
            "wind_gust_10_m_above_gnd",
            "angle_of_incidence",
            "zenith",
            "azimuth",
        ]

        # Get request data
        data = request.json
        input_features = np.array([data[col] for col in input_col]).reshape(1, -1)

        # Standardize input
        input_features_scaled = scaler_X.transform(input_features)

        # Make prediction
        prediction_scaled = model.predict(input_features_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]

        return jsonify({"predicted_energy_usage": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask server
if __name__ == "__main__":
    app.run(debug=True)
