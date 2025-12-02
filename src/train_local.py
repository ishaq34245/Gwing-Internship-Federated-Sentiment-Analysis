# src/train_local.py
# Useful to test training on one client without Flower
import torch
from transformers import DistilBertForSequenceClassification, AdamW
from src.utils import load_client_data, MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = load_client_data("../data/clients/client_0.csv", batch_size=8)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(1):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print("Epoch done")
