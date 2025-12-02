import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--raw", type=str, default="../data/raw.csv")
parser.add_argument("--clients", type=int, default=3)
parser.add_argument("--out_dir", type=str, default="../data/clients")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

df = pd.read_csv(args.raw)  # Expect columns: text,label

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# If there are many classes, ensure each client gets some samples
splits = np.array_split(df, args.clients)

for i, part in enumerate(splits):
    path = os.path.join(args.out_dir, f"client_{i}.csv")
    part.to_csv(path, index=False)
    print("Saved", path)
