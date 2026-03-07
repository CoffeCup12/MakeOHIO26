from flask import Flask, request, jsonify
import torch
from model import LineRatePredictor

# Load model
model = LineRatePredictor(1)
model.load_state_dict(torch.load("./model.pth", weights_only=True))
model.eval()

app = Flask(__name__)

@app.route("/analyze", methods=['POST'])
def analyze():
    try:
        # Read JSON data
        data = request.get_json()

        # Example JSON payload: {"delta_temp": [0.1, 0.2, 0.3]}
        delta_temps = data['delta_temp']

        # Convert to tensor
        in_feature = torch.tensor(delta_temps, dtype=torch.float32).unsqueeze(0)  # add batch dim

        # Forward pass
        with torch.no_grad():
            pred = model(in_feature)

        # Assuming output is probability, threshold at 0.5
        warning = (pred > 0.5).any().item()

        if warning:
            print("Warning")
            return jsonify({"status": "warning"})
        else:
            print("Normal")
            return jsonify({"status": "normal"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)