import numpy as np
import pandas as pd
import csv

np.random.seed(42)

def generate_temperature_series(length=5000):

    temps = []
    temp = 20

    for _ in range(length):

        # random drift
        temp += np.random.normal(0,0.5)

        # occasional heating event
        if np.random.rand() < 0.02:
            temp += np.random.uniform(3,8)

        # clamp realistic range
        temp = max(5, min(40,temp))

        temps.append(temp)

    return np.array(temps)

def create_dataset(temps, window=20, horizon=5):

    X = []
    y = []

    for i in range(len(temps) - window - horizon):

        seq = temps[i:i+window]

        future = temps[i+window:i+window+horizon]

        label = int(np.any(future > 30))

        X.append(seq)
        y.append(label)

    return np.array(X), np.array(y)

temps = generate_temperature_series(10000)

X, y = create_dataset(temps)

print(X.shape)
print(y.shape)
res = []

for i, el in enumerate(X):
    res.append(np.append(el, y[i]))

with open("./dataset.csv", "w", newline='') as f1:
    writer = csv.writer(f1, delimiter=",")
    writer.writerows(res)
