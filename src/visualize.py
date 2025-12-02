# visualize.py
import matplotlib.pyplot as plt
import os
import pandas as pd

CSV_PATH = os.path.join("checkpoints", "training_metrics.csv")
OUT_DIR = os.path.join("checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print("No metrics CSV found at", CSV_PATH, "- run server first to gather metrics.")
    raise SystemExit(1)

df = pd.read_csv(CSV_PATH)
rounds = df["round"].tolist()

# Plot train_loss if present
if "train_loss" in df.columns:
    plt.figure(figsize=(8,4))
    plt.plot(rounds, df["train_loss"].tolist(), marker='o')
    plt.xlabel("Round")
    plt.ylabel("Train Loss")
    plt.title("Train Loss per Round")
    plt.grid(True)
    outp = os.path.join(OUT_DIR, "training_loss.png")
    plt.savefig(outp, dpi=150)
    plt.close()
    print("Saved:", outp)

# Plot accuracy if present
if "accuracy" in df.columns:
    plt.figure(figsize=(8,4))
    plt.plot(rounds, df["accuracy"].tolist(), marker='o')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Round")
    plt.grid(True)
    outp = os.path.join(OUT_DIR, "training_accuracy.png")
    plt.savefig(outp, dpi=150)
    plt.close()
    print("Saved:", outp)
