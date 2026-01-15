# load_and_infer.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "prajjwal1/bert-tiny"
NUM_LABELS = 4
checkpoint = "checkpoints/aggregated_round_3.pt"  # choose the final round

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
state = torch.load(checkpoint, map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def predict(text):
    enc = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        pred = int(torch.argmax(torch.tensor(probs)).item())
    return pred, probs

if __name__ == "__main__":
    txt = "Example text to classify"
    print(predict(txt))
