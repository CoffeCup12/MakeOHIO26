from flask import Flask
from flask import request
import torch

from model import LineRatePredictor

model = LineRatePredictor(1)
model.load_state_dict(torch.load("./model.pth", weights_only=True))
model.eval()

app = Flask(__name__)

@app.route("/analyze", method = ['POST'])
def analyze():
    delta_temps = request.form['delta_temp']
    in_feature = torch.tensor(delta_temps).unsqueeze(0)

    pred = model(in_feature)

    if 1 in pred:
        print("warning")



