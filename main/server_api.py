# server_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "prajjwal1/bert-tiny"
NUM_LABELS = 4
CHECKPOINT = "checkpoints/aggregated_round_3.pt"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"), strict=False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

app = FastAPI()

class Req(BaseModel):
    text: str

@app.post("/predict")
def predict(req: Req):
    enc = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**enc).logits
        pred = int(logits.argmax(dim=-1).item())
    return {"prediction": pred}
