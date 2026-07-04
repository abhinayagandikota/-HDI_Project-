import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

artifacts_dir = os.path.join(os.path.dirname(__file__), "src", "artifacts")
model_path = os.path.join(artifacts_dir, "hdi_model.pkl")
scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
encoder_path = os.path.join(artifacts_dir, "encoder.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
le = pickle.load(open(encoder_path, "rb"))

FEATURES = ["life_expectancy", "mean_schooling_years", "expected_schooling_years", "gni_per_capita"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_data = {}
    if request.method == "POST":
        try:
            input_data = {
                "life_expectancy": float(request.form["life_expectancy"]),
                "mean_schooling_years": float(request.form["mean_schooling_years"]),
                "expected_schooling_years": float(request.form["expected_schooling_years"]),
                "gni_per_capita": float(request.form["gni_per_capita"]),
            }
            features = np.array([[input_data[f] for f in FEATURES]])
            features_scaled = scaler.transform(features)
            pred_encoded = model.predict(features_scaled)[0]
            prediction = le.inverse_transform([pred_encoded])[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction, input=input_data)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    try:
        features = np.array([[data[f] for f in FEATURES]])
        features_scaled = scaler.transform(features)
        pred_encoded = model.predict(features_scaled)[0]
        prediction = le.inverse_transform([pred_encoded])[0]
        return jsonify({"prediction": prediction, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
