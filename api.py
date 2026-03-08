from flask import Flask, request, jsonify
import torch
from model import LineRatePredictor

# Load model
model = LineRatePredictor(1)
model.load_state_dict(torch.load("./model4.pth", weights_only=True, map_location=torch.device('cpu')))
model.eval()

# warning flag
is_warning = False

app = Flask(__name__)

@app.route("/analyze", methods=['POST'])
def analyze():
    global is_warning
    try:
        
        data = request.get_json()
        delta_temps = data['delta_temp']

        in_feature = torch.tensor(delta_temps, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) 

        with torch.no_grad():
            pred = model(in_feature)

        is_warning = (pred > 0.8).any().item()

        if is_warning:
            print("Warning")
            return jsonify({"status": "warning"})
        else:
            print("Normal")
            return jsonify({"status": "normal"})

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 400
    
@app.route("/status", methods=["GET"])
def status():
    return jsonify({"warning": is_warning})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)