import numpy as np
import pandas as pd

np.random.seed(42)

# dataset parameters
num_samples = 5000
sampling_seconds = 10

# base values
wind_base = 3.0
temp_base = 22.0
hum_base = 40.0

wind = []
temp = []
hum = []
labels = []

anomaly_prob = 0.01
anomaly_duration = 30

i = 0
while i < num_samples:

    # decide whether to start anomaly
    if np.random.rand() < anomaly_prob and i < num_samples - anomaly_duration:

        # gradual buildup before anomaly
        for j in range(anomaly_duration):

            drift = j / anomaly_duration

            wind.append(wind_base + 5*drift + np.random.normal(0,0.2))
            temp.append(temp_base + 6*drift + np.random.normal(0,0.3))
            hum.append(hum_base + 20*drift + np.random.normal(0,1))

            if j > anomaly_duration * 0.8:
                labels.append(1)
            else:
                labels.append(0)

            i += 1

    else:
        wind.append(wind_base + np.random.normal(0,0.3))
        temp.append(temp_base + np.random.normal(0,0.4))
        hum.append(hum_base + np.random.normal(0,1.5))
        labels.append(0)
        i += 1


# timestamps
timestamps = pd.date_range(
    start="2026-01-01",
    periods=num_samples,
    freq=f"{sampling_seconds}s"
)

df = pd.DataFrame({
    "timestamp": timestamps,
    "wind": wind,
    "temperature": temp,
    "humidity": hum,
    "label": labels
})

df.to_csv("sensor_dataset.csv", index=False)

print("Dataset saved to sensor_dataset.csv")
print(df.head())