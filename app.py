from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import joblib
# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load trained model
try:
    model = joblib.load("RFModel.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None


# API route to predict energy usage
@app.route("/predict", methods=["POST"])
def predict():
    try:
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
        data = request.json  # Receive data from frontend
        input_features = np.array([data[col] for col in input_col]).reshape(1, -1)
        prediction = model.predict(input_features)
        return jsonify({"predicted_energy_usage": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})


# Run Flask server
if __name__ == "__main__":
    app.run(debug=True)
