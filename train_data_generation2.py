import numpy as np
import pandas as pd

np.random.seed(42)

# --------------------------------------------------
# Generate a realistic delta_temp sequence
# --------------------------------------------------

def generate_delta_temp_series(length=200):

    series = []
    value = np.random.uniform(15,25)

    for _ in range(length):

        mode = np.random.choice(
            ["stable","heat","cool","spike"],
            p=[0.5,0.25,0.2,0.05]
        )

        if mode == "stable":
            value += np.random.normal(0,0.5)

        elif mode == "heat":
            value += np.random.uniform(0.5,2)

        elif mode == "cool":
            value -= np.random.uniform(0.5,2)

        elif mode == "spike":
            value += np.random.uniform(5,10)

        value = np.clip(value,0,100)

        series.append(value)

    return np.array(series)


# --------------------------------------------------
# Create sliding window dataset
# --------------------------------------------------

def create_dataset(series, window=20, horizon=5, threshold=30):

    rows = []

    for i in range(len(series) - window - horizon):

        past = series[i:i+window]
        future = series[i+window:i+window+horizon]

        label = int(np.any(future > threshold))

        row = list(past) + [label]

        rows.append(row)

    return rows


# --------------------------------------------------
# Build full dataset
# --------------------------------------------------

def build_dataset(num_series=200):

    dataset = []

    for _ in range(num_series):

        series = generate_delta_temp_series(200)

        rows = create_dataset(series)

        dataset.extend(rows)

    return dataset


# --------------------------------------------------
# Generate dataset
# --------------------------------------------------

dataset = build_dataset()

print("Total samples:", len(dataset))


# --------------------------------------------------
# Save to CSV
# --------------------------------------------------

columns = [f"t{i}" for i in range(1,21)] + ["label"]

df = pd.DataFrame(dataset, columns=columns)

df.to_csv("delta_temp_dataset.csv", index=False)

print("Dataset saved to delta_temp_dataset.csv")