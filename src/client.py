# src/client.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import flwr as fl
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
from collections import Counter

from src.utils import load_client_data, get_numpy_parameters, set_model_params_from_numpy, MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--cid", type=int, required=True)
parser.add_argument("--data", type=str, default="../data/clients")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--local_epochs", type=int, default=1)
parser.add_argument("--max_batches", type=int, default=1000000)  # safety cap high by default
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, max_batches, label_map=None, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.max_batches = max_batches
        self.label_map = label_map
        self.class_weights = class_weights

        # Loss function (optionally weighted)
        if self.class_weights is not None:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        print("[CLIENT] Sending initial parameters to server (debug enabled)")
        return get_numpy_parameters(self.model)

    def fit(self, parameters, config):
        print("[CLIENT] Received training request from server")
        set_model_params_from_numpy(self.model, parameters)

        self.model.to(device)
        self.model.train()
        # Slightly higher LR for better convergence
        optimizer = AdamW(self.model.parameters(), lr=7e-5)

        batches = 0
        total_loss = 0.0
        total_batches = 0

        for epoch in range(args.local_epochs):
            print(f"[CLIENT] Epoch {epoch+1}/{args.local_epochs}")
            for batch in self.train_loader:
                if batches >= self.max_batches:
                    print("[CLIENT] Reached max batch limit, stopping early")
                    break
                batches += 1

                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).long()

                # remap labels if needed
                if self.label_map is not None:
                    labels_np = labels.cpu().numpy()
                    mapped = np.array([self.label_map[int(x)] for x in labels_np], dtype=np.int64)
                    labels = torch.tensor(mapped, dtype=torch.long, device=device)

                # guard
                max_label = int(labels.max().item())
                n_labels_model = int(self.model.config.num_labels)
                if max_label >= n_labels_model:
                    print(f"[CLIENT ERROR] Found label {max_label} >= model.num_labels ({n_labels_model})")
                    print("Unique labels in offending batch:", torch.unique(labels))
                    raise ValueError("Label out of bounds — check dataset labels or model.num_labels.")

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Use our (possibly weighted) loss
                loss = self.loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1

                if total_batches % 10 == 0:
                    print(f"[CLIENT] Batch {batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, total_batches)
        metrics = {"train_loss": float(avg_loss)}
        print(f"[CLIENT] Finished local training — avg_loss={avg_loss:.6f}")

        return get_numpy_parameters(self.model), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        print("[CLIENT] Received evaluation request from server")
        set_model_params_from_numpy(self.model, parameters)
        self.model.to(device)
        self.model.eval()

        # Use same class weights (if any), but sum reduction for total loss
        if self.class_weights is not None:
            loss_f = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction="sum")
        else:
            loss_f = torch.nn.CrossEntropyLoss(reduction="sum")

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # remap labels if needed
                if self.label_map is not None:
                    labels_np = labels.cpu().numpy()
                    mapped = np.array([self.label_map[int(x)] for x in labels_np], dtype=np.int64)
                    labels = torch.tensor(mapped, dtype=torch.long, device=device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_loss = float(loss_f(logits, labels).item())
                total_loss += batch_loss

                preds = logits.argmax(dim=-1)
                correct += int((preds == labels).sum().item())
                total += labels.size(0)

        avg_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        print(f"[CLIENT] Evaluation finished — avg_loss={avg_loss:.6f}, accuracy={accuracy:.4f}")

        return float(avg_loss), total, {"accuracy": float(accuracy)}


if __name__ == "__main__":
    client_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "clients", f"client_{args.cid}.csv")
    )

    print("Using client file:", client_file)

    train_loader = load_client_data(client_file, batch_size=args.batch_size)

    # -----------------------------
    # Determine labels and counts in the dataset
    # -----------------------------
    unique_labels = set()
    label_counts = Counter()

    try:
        ds = train_loader.dataset
        for i in range(len(ds)):
            item = ds[i]
            lbl = None
            if isinstance(item, dict):
                lbl = item.get("labels")
            else:
                if len(item) >= 2:
                    lbl = item[1]
            if lbl is None:
                continue
            if isinstance(lbl, torch.Tensor):
                lbl = int(lbl.item())
            lbl = int(lbl)
            unique_labels.add(lbl)
            label_counts[lbl] += 1
    except Exception as e:
        print("[WARN] Could not inspect dataset:", e)

    if len(unique_labels) == 0:
        print("[WARN] No labels found — defaulting to num_labels=2")
        num_labels = 2
        label_map = None
    else:
        sorted_labels = sorted(unique_labels)
        min_label, max_label = sorted_labels[0], sorted_labels[-1]
        num_unique = len(sorted_labels)
        print(f"[DATA] Unique labels detected: {sorted_labels}")
        print(f"[DATA] Label counts: {label_counts}")

        if min_label == 0 and max_label == num_unique - 1:
            num_labels = num_unique
            label_map = None
            print("[DATA] Labels are already 0-based and contiguous.")
        else:
            label_map = {orig: i for i, orig in enumerate(sorted_labels)}
            num_labels = num_unique
            print("[DATA] Remapping labels:", label_map)

    # -----------------------------
    # Compute class weights (simple inverse-frequency)
    # -----------------------------
    if len(label_counts) > 0:
        total_count = sum(label_counts.values())
        weights = []
        for i in range(num_labels):
            # If we remapped labels, map back to original label for counting
            if label_map is not None:
                inv_map = {v: k for k, v in label_map.items()}
                orig_label = inv_map.get(i, None)
                freq = label_counts.get(orig_label, 1)
            else:
                freq = label_counts.get(i, 1)

            # Simple inverse frequency weight
            w = total_count / (num_labels * freq)
            weights.append(w)

        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"[DATA] Computed class weights: {weights}")
    else:
        class_weights = None
        print("[DATA] Could not compute class weights; proceeding without weighting.")

    print(f"[MODEL] Loading model with num_labels={num_labels}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    ).to(device)

    client = FlowerClient(
        model,
        train_loader,
        args.max_batches,
        label_map=label_map,
        class_weights=class_weights,
    )

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
