# evaluate_saved.py
import os
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.utils import MODEL_NAME, detect_num_labels

# -----------------------------
# Config
# -----------------------------
TEST_CSV = os.path.join("data", "test.csv")
CHECKPOINT_DIR = os.path.join("checkpoints", "final_model")
BATCH_SIZE = 16
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load test CSV
# -----------------------------
if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

df = pd.read_csv(TEST_CSV)

if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("test.csv must have 'text' and 'label' columns")

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()
total = len(labels)

print(f"[EVAL] Loaded {total} test examples from {TEST_CSV}")

labels_tensor = torch.tensor(labels, dtype=torch.long)

# -----------------------------
# Detect num_labels
# -----------------------------
num_labels = detect_num_labels(clients_dir=os.path.join("data", "clients"))
print(f"[EVAL] Detected num_labels = {num_labels}")

# -----------------------------
# Load federated model
# -----------------------------
if os.path.isdir(CHECKPOINT_DIR):
    load_dir = CHECKPOINT_DIR
    print(f"[EVAL] Loading model from {load_dir}")
else:
    load_dir = MODEL_NAME
    print(f"[EVAL] final_model not found. Loading base model {MODEL_NAME}")

# Load model & tokenizer from load_dir
model = AutoModelForSequenceClassification.from_pretrained(load_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(load_dir)

model.eval()

# -----------------------------
# Batched evaluation
# -----------------------------
correct = 0
total_examples = total

with torch.no_grad():
    for start in range(0, total_examples, BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_texts = texts[start:end]
        batch_labels = labels_tensor[start:end].to(device)

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = logits.argmax(dim=-1)

        correct += int((preds == batch_labels).sum().item())

accuracy = correct / max(1, total_examples)

print("\n====================================")
print(f" TEST SAMPLES     = {total_examples}")
print(f" CORRECT PREDICTS = {correct}")
print(f" FINAL ACCURACY   = {accuracy:.4f}")
print("====================================\n")
